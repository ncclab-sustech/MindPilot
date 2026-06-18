"""Unified visual feature-map extraction for MindPilot encoding.

This replaces the per-model extraction scripts with one CLI:

    python extract_feature_maps.py --dnn alexnet
    python extract_feature_maps.py --dnn cornet_s --layer all
    python extract_feature_maps.py --dnn dino_vit_b_16 --dino_repo /path/to/dino

The output layout remains compatible with ``feature_maps_pca.py``:

    $PROJECT_DIR/dnn_feature_maps_{dnn}/full_feature_maps/{dnn}/pretrained-{bool}/{split}
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision import transforms as trn
from tqdm import tqdm

from paths import (
    ALEXNET_WEIGHT,
    CORNET_S_WEIGHT,
    DINO_VITB16_WEIGHT,
    DINOV2_VITB14_WEIGHT,
    IMAGE_SET_DIR,
    MOCO_WEIGHT,
    OPENCLIP_VITB32_WEIGHT,
    PROJECT_DIR,
    RESNET50_WEIGHT,
    VIT_B_16_HF_DIR,
    VIT_B_32_HF_DIR,
    str2bool,
)


SUPPORTED_DNNS = (
    "alexnet",
    "resnet50",
    "cornet_s",
    "moco",
    "vit_b_32",
    "openclip_vit_b_32",
    "dino_vit_b_16",
    "dino2_vit_b_14",
    "synclr_vit_b_16",
)


def _torch_load(path, **kwargs):
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(path, **kwargs)


def set_reproducible(seed, deterministic):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def imagenet_preprocess():
    return trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def dino_preprocess():
    return trn.Compose([
        trn.Resize(256, interpolation=3),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class AlexNetFeatures(nn.Module):
    conv_layers = [
        "conv1", "ReLU1", "maxpool1", "conv2", "ReLU2", "maxpool2",
        "conv3", "ReLU3", "conv4", "ReLU4", "conv5", "ReLU5", "maxpool5",
    ]
    fc_layers = ["Dropout6", "fc6", "ReLU6", "Dropout7", "fc7", "ReLU7", "fc8"]

    def __init__(self, pretrained):
        super().__init__()
        self.feat_list = ["maxpool1", "maxpool2", "ReLU3", "ReLU4", "maxpool5", "ReLU6", "ReLU7", "fc8"]
        model = models.alexnet(weights=None)
        if pretrained:
            model.load_state_dict(_torch_load(ALEXNET_WEIGHT, map_location="cpu", weights_only=True))
        self.features = model.features
        self.classifier = model.classifier
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        outputs = []
        for name, layer in self.features._modules.items():
            x = layer(x)
            if self.conv_layers[int(name)] in self.feat_list:
                outputs.append(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        for name, layer in self.classifier._modules.items():
            x = layer(x)
            if self.fc_layers[int(name)] in self.feat_list:
                outputs.append(x)
        return outputs


class ResNetFeatures(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.feat_list = ["block1", "block2", "block3", "block4", "fc"]
        self.features = {}
        hook_map = {
            "block1": self.model.layer1,
            "block2": self.model.layer2,
            "block3": self.model.layer3,
            "block4": self.model.layer4,
            "fc": self.model.fc,
        }
        self.hooks = [
            module.register_forward_hook(self._store(name))
            for name, module in hook_map.items()
        ]

    def _store(self, name):
        def hook(_module, _inputs, output):
            self.features[name] = output.detach()
        return hook

    def forward(self, x):
        self.features = {}
        self.model(x)
        return [self.features[name] for name in self.feat_list]


class HookedFeatures(nn.Module):
    def __init__(self, model, feat_list, forward_fn=None, postprocess=None):
        super().__init__()
        self.model = model
        self.feat_list = list(feat_list)
        self.forward_fn = forward_fn
        self.postprocess = postprocess
        self.features = {}
        modules = dict(self.model.named_modules())
        missing = [name for name in self.feat_list if name not in modules and name != "pooler"]
        if missing:
            raise ValueError(f"Missing feature layers in model: {missing}")
        self.hooks = [
            modules[name].register_forward_hook(self._store(name))
            for name in self.feat_list
            if name in modules
        ]

    def _store(self, name):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self.features[name] = output.detach()
        return hook

    def forward(self, x):
        self.features = {}
        if self.forward_fn is None:
            outputs = self.model(x)
        else:
            outputs = self.forward_fn(self.model, x)
        if "pooler" in self.feat_list and hasattr(outputs, "pooler_output"):
            self.features["pooler"] = outputs.pooler_output.detach()
        result = [(name, self.features[name]) for name in self.feat_list if name in self.features]
        if self.postprocess is not None:
            return [self.postprocess(name, feat) for name, feat in result]
        result = [feat for _name, feat in result]
        return result


class CornetFeatures(nn.Module):
    def __init__(self, model, selected_layers):
        super().__init__()
        self.model = model
        self.feat_list = list(selected_layers)
        self.features = {}
        module = self.model.module if hasattr(self.model, "module") else self.model
        self.hooks = []
        for layer_name in self.feat_list:
            model_layer = getattr(getattr(module, layer_name), "output")
            self.hooks.append(model_layer.register_forward_hook(self._store(layer_name)))

    def _store(self, name):
        def hook(_module, _inputs, output):
            output = output.detach()
            self.features[name] = output.reshape(output.shape[0], -1).unsqueeze(0)
        return hook

    def forward(self, x):
        self.features = {}
        self.model(x)
        return [self.features[name] for name in self.feat_list]


def build_resnet50(pretrained):
    model = models.resnet50(weights=None)
    if pretrained:
        model.load_state_dict(_torch_load(RESNET50_WEIGHT, map_location="cpu", weights_only=True))
    return ResNetFeatures(model), lambda image: imagenet_preprocess()(image).unsqueeze(0)


def build_moco(pretrained):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(in_features=2048, out_features=128)
    if pretrained:
        checkpoint = _torch_load(MOCO_WEIGHT, map_location="cpu")
        state_dict = {
            key.replace("module.encoder_q.", ""): value
            for key, value in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state_dict)
    return ResNetFeatures(model), lambda image: imagenet_preprocess()(image).unsqueeze(0)


def build_cornet_s(pretrained, layer):
    import cornet

    map_location = None if torch.cuda.is_available() else "cpu"
    model_fn = getattr(cornet, "cornet_s")
    if pretrained and CORNET_S_WEIGHT and os.path.isfile(CORNET_S_WEIGHT):
        model = model_fn(pretrained=False, map_location=map_location)
        ckpt_data = _torch_load(CORNET_S_WEIGHT, map_location=map_location, weights_only=False)
        model.load_state_dict(ckpt_data["state_dict"] if "state_dict" in ckpt_data else ckpt_data)
    else:
        model = model_fn(pretrained=pretrained, map_location=map_location)
    selected_layers = ["V1", "V2", "V4", "IT", "decoder"] if layer == "all" else [layer]
    return CornetFeatures(model, selected_layers), lambda image: imagenet_preprocess()(image).unsqueeze(0)


def build_vit_b_32(pretrained):
    from transformers import ViTConfig, ViTImageProcessor, ViTModel

    model = ViTModel.from_pretrained(VIT_B_32_HF_DIR, local_files_only=True) if pretrained else ViTModel(ViTConfig())
    image_processor = (
        ViTImageProcessor.from_pretrained(VIT_B_32_HF_DIR, local_files_only=True)
        if pretrained else ViTImageProcessor()
    )
    layers = ["encoder.layer.3", "encoder.layer.6", "encoder.layer.9", "encoder.layer.11"]
    extractor = HookedFeatures(model, layers)
    return extractor, lambda image: image_processor(image, return_tensors="pt")["pixel_values"]


def build_synclr_vit_b_16(pretrained):
    from transformers import ViTConfig, ViTImageProcessor, ViTModel

    if pretrained:
        model = ViTModel.from_pretrained(VIT_B_16_HF_DIR, local_files_only=True)
        image_processor = ViTImageProcessor.from_pretrained(VIT_B_16_HF_DIR, local_files_only=True)
    else:
        config = ViTConfig()
        config.patch_size = 16
        model = ViTModel(config)
        image_processor = ViTImageProcessor()

    def postprocess(name, feat):
        if name != "pooler" and feat.dim() == 3:
            return feat.mean(dim=1)
        return feat

    layers = ["encoder.layer.3", "encoder.layer.6", "encoder.layer.9", "encoder.layer.11", "pooler"]
    extractor = HookedFeatures(model, layers, postprocess=postprocess)
    return extractor, lambda image: image_processor(image, return_tensors="pt")["pixel_values"]


def build_openclip_vit_b_32(pretrained):
    import open_clip
    from open_clip import load_checkpoint

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
    if pretrained:
        original_torch_load = torch.load

        def trusted_torch_load(*load_args, **load_kwargs):
            load_kwargs.setdefault("weights_only", False)
            return original_torch_load(*load_args, **load_kwargs)

        torch.load = trusted_torch_load
        try:
            load_checkpoint(model, OPENCLIP_VITB32_WEIGHT)
        finally:
            torch.load = original_torch_load
    layers = [
        "visual.transformer.resblocks.3",
        "visual.transformer.resblocks.6",
        "visual.transformer.resblocks.9",
        "visual.transformer.resblocks.11",
    ]
    extractor = HookedFeatures(model, layers, forward_fn=lambda m, x: m.encode_image(x))
    return extractor, lambda image: preprocess(image).unsqueeze(0)


def build_dino_vit_b_16(pretrained, dino_repo):
    source = "local" if os.path.isdir(dino_repo) else "github"
    model = torch.hub.load(dino_repo, "dino_vitb16", source=source, pretrained=False)
    if pretrained:
        model.load_state_dict(_torch_load(DINO_VITB16_WEIGHT, map_location="cpu", weights_only=True), strict=True)
    layers = ["blocks.3", "blocks.6", "blocks.9", "blocks.11"]
    return HookedFeatures(model, layers), lambda image: dino_preprocess()(image).unsqueeze(0)


def build_dino2_vit_b_14(pretrained, dinov2_repo):
    source = "local" if os.path.isdir(dinov2_repo) else "github"
    model = torch.hub.load(dinov2_repo, "dinov2_vitb14", source=source, pretrained=False)
    if pretrained:
        model.load_state_dict(_torch_load(DINOV2_VITB14_WEIGHT, map_location="cpu", weights_only=True), strict=True)
    layers = ["blocks.3", "blocks.6", "blocks.9", "blocks.11"]
    return HookedFeatures(model, layers), lambda image: dino_preprocess()(image).unsqueeze(0)


def build_extractor(args):
    if args.dnn == "alexnet":
        return AlexNetFeatures(args.pretrained), lambda image: imagenet_preprocess()(image).unsqueeze(0)
    if args.dnn == "resnet50":
        return build_resnet50(args.pretrained)
    if args.dnn == "moco":
        return build_moco(args.pretrained)
    if args.dnn == "cornet_s":
        return build_cornet_s(args.pretrained, args.layer)
    if args.dnn == "vit_b_32":
        return build_vit_b_32(args.pretrained)
    if args.dnn == "synclr_vit_b_16":
        return build_synclr_vit_b_16(args.pretrained)
    if args.dnn == "openclip_vit_b_32":
        return build_openclip_vit_b_32(args.pretrained)
    if args.dnn == "dino_vit_b_16":
        return build_dino_vit_b_16(args.pretrained, args.dino_repo)
    if args.dnn == "dino2_vit_b_14":
        return build_dino2_vit_b_14(args.pretrained, args.dinov2_repo)
    raise ValueError(f"Unsupported dnn: {args.dnn}")


def iter_images(split_dir):
    split_dir = Path(split_dir)
    folders = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not folders:
        folders = [split_dir]
    for folder in folders:
        images = sorted(
            path for path in folder.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg"}
        )
        yield folder.name, images


def extract_split(args, model, preprocess, device, split):
    split_dir = Path(args.image_set_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Image split directory not found: {split_dir}")

    save_dir = (
        Path(args.project_dir) / f"dnn_feature_maps_{args.dnn}" /
        "full_feature_maps" / args.dnn / f"pretrained-{args.pretrained}" / split
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    for partition, image_list in iter_images(split_dir):
        print(f"Processing partition: {partition}")
        for index, image_path in enumerate(tqdm(image_list, desc=f"Extracting {args.dnn} features for {partition}")):
            img = Image.open(image_path).convert("RGB")
            input_img = preprocess(img).to(device)
            with torch.no_grad():
                outputs = model(input_img)
            feats = {
                layer_name: feat.detach().cpu().numpy()
                for layer_name, feat in zip(model.feat_list, outputs)
            }
            file_name = f"{partition}_{index + 1:07d}"
            np.save(save_dir / file_name, feats)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Extract visual model feature maps.")
    parser.add_argument("--dnn", default="alexnet", choices=SUPPORTED_DNNS)
    parser.add_argument("--pretrained", default=True, type=str2bool)
    parser.add_argument("--layer", default="all", choices=["all", "V1", "V2", "V4", "IT", "decoder"])
    parser.add_argument("--project_dir", default=PROJECT_DIR, type=str)
    parser.add_argument("--image_set_dir", default=IMAGE_SET_DIR, type=str)
    parser.add_argument("--partitions", nargs="+", default=["test_images", "training_images"])
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--deterministic", default=False, type=str2bool)
    parser.add_argument(
        "--dino_repo",
        default=os.environ.get("DINO_REPO", "facebookresearch/dino:main"),
        type=str,
        help="Local DINO repo path or torch.hub repo spec.",
    )
    parser.add_argument(
        "--dinov2_repo",
        default=os.environ.get("DINOV2_REPO", "facebookresearch/dinov2"),
        type=str,
        help="Local DINOv2 repo path or torch.hub repo spec.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f">>> Extract feature maps {args.dnn} <<<")
    print("\nInput arguments:")
    for key, val in vars(args).items():
        print(f"{key:16} {val}")

    set_reproducible(args.seed, args.deterministic)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, preprocess = build_extractor(args)
    model.to(device)
    model.eval()

    for split in args.partitions:
        extract_split(args, model, preprocess, device, split)


if __name__ == "__main__":
    main()
