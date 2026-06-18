"""EEG dataset utilities used by the ATMS retrieval model.

This module intentionally avoids import-time network, proxy, checkpoint, or GPU
side effects. Configure local assets with constructor arguments or environment
variables:

- ``IMAGE_SET_DIR``: directory containing ``training_images/`` and
  ``test_images/``.
- ``FEATURE_CACHE_DIR``: directory for cached OpenCLIP text/image features.
- ``OPENCLIP_WEIGHT``: optional local OpenCLIP checkpoint.
- ``OPENCLIP_PRETRAINED``: OpenCLIP pretrained tag when no local checkpoint is
  supplied. Defaults to ``laion2b_s32b_b79k``.
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

try:
	import open_clip
except ImportError:  # pragma: no cover - handled when a model is requested.
	open_clip = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_IMAGE_SET_DIR = Path("data") / "images_set"
DEFAULT_FEATURE_CACHE_DIR = Path("data") / "clip_features"
DEFAULT_MODEL_TYPE = "ViT-H-14"

_MODEL_CACHE = {}


def _as_path(path_like):
	return Path(path_like).expanduser()


def _resolve_repo_path(path_like):
	path = _as_path(path_like)
	if path.is_absolute():
		return path
	return REPO_ROOT / path


def _get_device(device=None):
	if device is not None:
		return torch.device(device)
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_openclip(model_type=DEFAULT_MODEL_TYPE, model_weights_path=None,
	device=None, pretrained=None):
	if open_clip is None:
		raise ImportError(
			"open_clip is required for EEGDataset feature encoding. "
			"Install it or precompute and cache features in FEATURE_CACHE_DIR."
		)

	device = _get_device(device)
	model_weights_path = model_weights_path or os.environ.get("OPENCLIP_WEIGHT")
	pretrained = pretrained or os.environ.get("OPENCLIP_PRETRAINED", "laion2b_s32b_b79k")
	cache_key = (model_type, str(model_weights_path or ""), str(pretrained or ""), str(device))
	if cache_key in _MODEL_CACHE:
		return _MODEL_CACHE[cache_key]

	if model_weights_path:
		model_weights_path = _as_path(model_weights_path)
		if not model_weights_path.exists():
			raise FileNotFoundError(
				f"OpenCLIP checkpoint not found: {model_weights_path}. "
				"Set OPENCLIP_WEIGHT to an existing file or unset it to use "
				"OPENCLIP_PRETRAINED."
			)
		vlmodel, preprocess_train, _ = open_clip.create_model_and_transforms(
			model_name=model_type, pretrained=None, precision="fp32", device=device
		)
		state_dict = torch.load(model_weights_path, map_location=device)
		vlmodel.load_state_dict(state_dict)
	else:
		vlmodel, preprocess_train, _ = open_clip.create_model_and_transforms(
			model_name=model_type, pretrained=pretrained, precision="fp32", device=device
		)

	vlmodel.to(device)
	vlmodel.eval()
	tokenizer = open_clip.get_tokenizer(model_type)
	_MODEL_CACHE[cache_key] = (vlmodel, preprocess_train, tokenizer)
	return _MODEL_CACHE[cache_key]


class EEGDataset(Dataset):
	"""
	Dataset for leave-one-subject EEG retrieval training/evaluation.

	Expected EEG directory layout:

	```
	data_path/
	  sub-01/preprocessed_eeg_training.npy
	  sub-01/preprocessed_eeg_test.npy
	  ...
	```
	"""

	def __init__(
		self,
		data_path,
		exclude_subject=None,
		subjects=None,
		train=True,
		time_window=(0, 1.0),
		classes=None,
		pictures=None,
		val_size=None,
		image_set_dir=None,
		feature_cache_dir=None,
		model_type=DEFAULT_MODEL_TYPE,
		model_weights_path=None,
		openclip_pretrained=None,
		device=None,
	):
		self.data_path = _resolve_repo_path(data_path)
		self.train = train
		self.subject_list = sorted(p.name for p in self.data_path.iterdir() if p.is_dir())
		self.subjects = self.subject_list if subjects is None else subjects
		self.n_sub = len(self.subjects)
		self.time_window = time_window
		self.n_cls = 1654 if train else 200
		self.classes = classes
		self.pictures = pictures
		self.exclude_subject = exclude_subject
		self.val_size = val_size
		self.device = _get_device(device)
		self.model_type = model_type
		self.model_weights_path = model_weights_path
		self.openclip_pretrained = openclip_pretrained

		image_set_dir = image_set_dir or os.environ.get("IMAGE_SET_DIR", DEFAULT_IMAGE_SET_DIR)
		feature_cache_dir = feature_cache_dir or os.environ.get(
			"FEATURE_CACHE_DIR", DEFAULT_FEATURE_CACHE_DIR
		)
		self.image_set_dir = _resolve_repo_path(image_set_dir)
		self.img_directory_training = self.image_set_dir / "training_images"
		self.img_directory_test = self.image_set_dir / "test_images"
		self.feature_cache_dir = _resolve_repo_path(feature_cache_dir)
		self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

		missing_subjects = [sub for sub in self.subjects if sub not in self.subject_list]
		if missing_subjects:
			raise ValueError(
				f"Missing subjects under {self.data_path}: {missing_subjects}. "
				f"Available subjects: {self.subject_list}"
			)

		self.vlmodel, self.preprocess_train, self.tokenizer = _load_openclip(
			model_type=self.model_type,
			model_weights_path=self.model_weights_path,
			device=self.device,
			pretrained=self.openclip_pretrained,
		)

		self.data, self.labels, self.text, self.img = self.load_data()
		self.data = self.extract_eeg(self.data, self.time_window)

		if self.classes is None and self.pictures is None:
			features_filename = self.feature_cache_dir / (
				f"{self.model_type}_features_train.pt"
				if self.train
				else f"{self.model_type}_features_test.pt"
			)
			if features_filename.exists():
				saved_features = torch.load(features_filename, map_location="cpu")
				self.text_features = saved_features["text_features"]
				self.img_features = saved_features["img_features"]
			else:
				self.text_features = self.Textencoder(self.text)
				self.img_features = self.ImageEncoder(self.img)
				torch.save(
					{
						"text_features": self.text_features.cpu(),
						"img_features": self.img_features.cpu(),
					},
					features_filename,
				)
		else:
			self.text_features = self.Textencoder(self.text)
			self.img_features = self.ImageEncoder(self.img)

	def load_data(self):
		data_list = []
		label_list = []
		texts = []
		images = []

		img_directory = self.img_directory_training if self.train else self.img_directory_test
		if not img_directory.is_dir():
			raise FileNotFoundError(
				f"Image directory not found: {img_directory}. "
				"Set IMAGE_SET_DIR or pass image_set_dir to EEGDataset."
			)

		dirnames = sorted(
			d.name for d in img_directory.iterdir() if d.is_dir()
		)
		if self.classes is not None:
			dirnames = [dirnames[i] for i in self.classes]

		for dirname in dirnames:
			try:
				idx = dirname.index("_")
				description = dirname[idx + 1 :]
			except ValueError:
				print(f"Skipped: {dirname} due to no '_' found.")
				continue
			texts.append(f"This picture is {description}")

		all_folders = sorted(d.name for d in img_directory.iterdir() if d.is_dir())
		if self.classes is not None and self.pictures is not None:
			for class_idx, pic_idx in zip(self.classes, self.pictures):
				if class_idx < len(all_folders):
					folder_path = img_directory / all_folders[class_idx]
					all_images = sorted(
						img for img in folder_path.iterdir()
						if img.suffix.lower() in {".png", ".jpg", ".jpeg"}
					)
					if pic_idx < len(all_images):
						images.append(str(all_images[pic_idx]))
		elif self.classes is not None:
			for class_idx in self.classes:
				if class_idx < len(all_folders):
					folder_path = img_directory / all_folders[class_idx]
					all_images = sorted(
						img for img in folder_path.iterdir()
						if img.suffix.lower() in {".png", ".jpg", ".jpeg"}
					)
					images.extend(str(img) for img in all_images)
		else:
			for folder in all_folders:
				folder_path = img_directory / folder
				all_images = sorted(
					img for img in folder_path.iterdir()
					if img.suffix.lower() in {".png", ".jpg", ".jpeg"}
				)
				images.extend(str(img) for img in all_images)

		print("self.subjects", self.subjects)
		print("exclude_subject", self.exclude_subject)

		times = None
		ch_names = None
		for subject in self.subjects:
			if self.train:
				if subject == self.exclude_subject:
					continue
				file_name = "preprocessed_eeg_training.npy"
				file_path = self.data_path / subject / file_name
				data = np.load(file_path, allow_pickle=True)
				preprocessed_eeg_data = torch.from_numpy(
					data["preprocessed_eeg_data"]
				).float().detach()
				times = torch.from_numpy(data["times"]).detach()[50:]
				ch_names = data["ch_names"]
				n_classes = 1654
				samples_per_class = 10

				if self.classes is not None and self.pictures is not None:
					for c, p in zip(self.classes, self.pictures):
						start_index = c * 1 + p
						if start_index < len(preprocessed_eeg_data):
							data_list.append(preprocessed_eeg_data[start_index : start_index + 1])
							label_list.append(torch.full((1,), c, dtype=torch.long).detach())
				elif self.classes is not None:
					for c in self.classes:
						start_index = c * samples_per_class
						data_list.append(
							preprocessed_eeg_data[start_index : start_index + samples_per_class]
						)
						label_list.append(
							torch.full((samples_per_class,), c, dtype=torch.long).detach()
						)
				else:
					for i in range(n_classes):
						start_index = i * samples_per_class
						data_list.append(
							preprocessed_eeg_data[start_index : start_index + samples_per_class]
						)
						label_list.append(
							torch.full((samples_per_class,), i, dtype=torch.long).detach()
						)
			else:
				if subject == self.exclude_subject or self.exclude_subject is None:
					file_name = "preprocessed_eeg_test.npy"
					file_path = self.data_path / subject / file_name
					data = np.load(file_path, allow_pickle=True)
					preprocessed_eeg_data = torch.from_numpy(
						data["preprocessed_eeg_data"]
					).float().detach()
					times = torch.from_numpy(data["times"]).detach()[50:]
					ch_names = data["ch_names"]
					n_classes = 200
					samples_per_class = 1

					for i in range(n_classes):
						if self.classes is not None and i not in self.classes:
							continue
						start_index = i * samples_per_class
						preprocessed_eeg_data_class = preprocessed_eeg_data[
							start_index : start_index + samples_per_class
						]
						labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
						preprocessed_eeg_data_class = torch.mean(
							preprocessed_eeg_data_class.squeeze(0), 0
						)
						data_list.append(preprocessed_eeg_data_class)
						label_list.append(labels)

		if not data_list:
			raise ValueError(
				"No EEG samples were loaded. Check data_path, subjects, "
				"exclude_subject, classes, and train/test settings."
			)

		if self.train:
			data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])
		else:
			data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)

		label_tensor = torch.cat(label_list, dim=0)
		if self.train:
			label_tensor = label_tensor.repeat_interleave(4)
			if self.classes is not None:
				unique_values = []
				for value in label_tensor.numpy():
					if value not in unique_values:
						unique_values.append(value)
				mapping = {val.item(): index for index, val in enumerate(torch.tensor(unique_values))}
				label_tensor = torch.tensor(
					[mapping[val.item()] for val in label_tensor], dtype=torch.long
				)

		self.times = times
		self.ch_names = ch_names
		print(
			f"Data tensor shape: {data_tensor.shape}, "
			f"label tensor shape: {label_tensor.shape}, "
			f"text length: {len(texts)}, image length: {len(images)}"
		)
		return data_tensor, label_tensor, texts, images

	def extract_eeg(self, eeg_data, time_window):
		start, end = time_window
		indices = (self.times >= start) & (self.times <= end)
		return eeg_data[..., indices]

	def Textencoder(self, text):
		text_inputs = self.tokenizer(text).to(self.device)
		with torch.no_grad():
			text_features = self.vlmodel.encode_text(text_inputs)
		return F.normalize(text_features, dim=-1).detach()

	def ImageEncoder(self, images):
		batch_size = 20
		image_features_list = []
		for i in range(0, len(images), batch_size):
			batch_images = images[i : i + batch_size]
			image_inputs = torch.stack(
				[self.preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]
			).to(self.device)
			with torch.no_grad():
				batch_image_features = self.vlmodel.encode_image(image_inputs)
				batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)
			image_features_list.append(batch_image_features)
		return torch.cat(image_features_list, dim=0)

	def __getitem__(self, index):
		x = self.data[index]
		label = self.labels[index]
		modal = "eeg"
		if self.pictures is None:
			if self.classes is None:
				index_n_sub_train = self.n_cls * 10 * 4
				index_n_sub_test = self.n_cls * 1 * 80
			else:
				index_n_sub_test = len(self.classes) * 1 * 80
				index_n_sub_train = len(self.classes) * 10 * 4
			if self.train:
				text_index = (index % index_n_sub_train) // (10 * 4)
				img_index = (index % index_n_sub_train) // 4
			else:
				text_index = index % index_n_sub_test
				img_index = index % index_n_sub_test
		else:
			if self.classes is None:
				index_n_sub_train = self.n_cls * 1 * 4
				index_n_sub_test = self.n_cls * 1 * 80
			else:
				index_n_sub_test = len(self.classes) * 1 * 80
				index_n_sub_train = len(self.classes) * 1 * 4
			if self.train:
				text_index = (index % index_n_sub_train) // (1 * 4)
				img_index = (index % index_n_sub_train) // 4
			else:
				text_index = index % index_n_sub_test
				img_index = index % index_n_sub_test

		text = self.text[text_index]
		img = self.img[img_index]
		text_features = self.text_features[text_index]
		img_features = self.img_features[img_index]
		return modal, x, label, text, text_features, img, img_features, -1

	def __len__(self):
		return self.data.shape[0]


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Smoke-test EEGDataset loading.")
	parser.add_argument("--data_path", required=True, help="Directory containing sub-XX EEG folders.")
	parser.add_argument("--subject", default="sub-01")
	parser.add_argument("--image_set_dir", default=None)
	parser.add_argument("--feature_cache_dir", default=None)
	parser.add_argument("--device", default=None)
	args = parser.parse_args()

	train_dataset = EEGDataset(
		args.data_path,
		subjects=[args.subject],
		train=True,
		image_set_dir=args.image_set_dir,
		feature_cache_dir=args.feature_cache_dir,
		device=args.device,
	)
	test_dataset = EEGDataset(
		args.data_path,
		subjects=[args.subject],
		train=False,
		image_set_dir=args.image_set_dir,
		feature_cache_dir=args.feature_cache_dir,
		device=args.device,
	)
	train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
	print(f"train batches: {len(train_loader)}, test batches: {len(test_loader)}")
