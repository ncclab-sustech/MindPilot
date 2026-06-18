#!/usr/bin/env python3
"""Download or stage all pretrained visual model weights for MindPilot encoding."""

import os
import shutil
import sys
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_WEIGHTS_DIR = SCRIPT_DIR
DNN_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'dnn_feature_maps_extraction')
TORCH_CACHE = os.path.expanduser('~/.cache/torch/hub/checkpoints')
HF_DIR = os.path.join(PRETRAIN_WEIGHTS_DIR, 'huggingface')

DOWNLOADS = {
	'alexnet-owt-7be5be79.pth':
		'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
	'resnet50-0676ba61.pth':
		'https://download.pytorch.org/models/resnet50-0676ba61.pth',
	'dino_vitbase16_pretrain.pth':
		'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
	'dinov2_vitb14_pretrain.pth':
		'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
	'openclip_vit_b_32_openai.pt':
		'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt',
}

COPY_SOURCES = {
	'moco_v1_200ep_pretrain.pth.tar': os.path.join(DNN_DIR, 'moco_v1_200ep_pretrain.pth.tar'),
	'cornet_s-1d3f7974.pth': os.path.join(DNN_DIR, 'cornet_s-1d3f7974.pth'),
}

HF_MODELS = {
	'google-vit-base-patch32-224-in21k': 'google/vit-base-patch32-224-in21k',
	'google-vit-base-patch16-224-in21k': 'google/vit-base-patch16-224-in21k',
}


def download_file(url, dest):
	if os.path.isfile(dest):
		print(f'[skip] {dest} already exists')
		return
	print(f'[download] {url} -> {dest}')
	os.makedirs(os.path.dirname(dest), exist_ok=True)
	tmp = dest + '.part'
	urllib.request.urlretrieve(url, tmp)
	os.replace(tmp, dest)
	print(f'[done] {dest}')


def copy_or_link(src, dest):
	if os.path.isfile(dest):
		print(f'[skip] {dest} already exists')
		return
	if not os.path.isfile(src):
		print(f'[warn] source missing: {src}')
		return
	os.makedirs(os.path.dirname(dest), exist_ok=True)
	try:
		os.link(src, dest)
		print(f'[link] {src} -> {dest}')
	except OSError:
		shutil.copy2(src, dest)
		print(f'[copy] {src} -> {dest}')


def stage_from_torch_cache(filename):
	src = os.path.join(TORCH_CACHE, filename)
	dest = os.path.join(PRETRAIN_WEIGHTS_DIR, filename)
	if os.path.isfile(dest):
		print(f'[skip] {dest} already exists')
		return
	if os.path.isfile(src):
		copy_or_link(src, dest)


def download_hf_models():
	from huggingface_hub import snapshot_download

	os.makedirs(HF_DIR, exist_ok=True)
	for local_name, repo_id in HF_MODELS.items():
		dest = os.path.join(HF_DIR, local_name)
		if os.path.isdir(dest) and os.listdir(dest):
			print(f'[skip] HF model already present: {dest}')
			continue
		print(f'[hf] downloading {repo_id} -> {dest}')
		snapshot_download(
			repo_id=repo_id,
			local_dir=dest,
			local_dir_use_symlinks=False,
		)
		print(f'[done] {dest}')


def main():
	os.makedirs(PRETRAIN_WEIGHTS_DIR, exist_ok=True)

	for filename, src in COPY_SOURCES.items():
		copy_or_link(src, os.path.join(PRETRAIN_WEIGHTS_DIR, filename))

	stage_from_torch_cache('alexnet-owt-7be5be79.pth')
	stage_from_torch_cache('resnet50-0676ba61.pth')

	for filename, url in DOWNLOADS.items():
		dest = os.path.join(PRETRAIN_WEIGHTS_DIR, filename)
		if os.path.isfile(dest):
			print(f'[skip] {dest} already exists')
			continue
		download_file(url, dest)

	download_hf_models()
	print('\nAll weights staged under:', PRETRAIN_WEIGHTS_DIR)


if __name__ == '__main__':
	main()
