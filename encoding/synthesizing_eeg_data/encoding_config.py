"""Shared experiment configuration for MindPilot EEG encoding."""

import argparse
import os
import shlex
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ENCODING_DIR = SCRIPT_DIR.parent
MINDPILOT_DIR = ENCODING_DIR.parent
VISUAL_ROOT = MINDPILOT_DIR.parent
DATASET_ROOT = Path(os.environ.get('DATASET_ROOT', VISUAL_ROOT / 'dataset'))


def _path_from_env(name, default):
	return str(Path(os.environ.get(name, default)).expanduser())


PROJECT_DIR = _path_from_env('PROJECT_DIR', ENCODING_DIR / 'DNNs')
PCA_SCRIPT_DIR = _path_from_env(
	'PCA_SCRIPT_DIR', ENCODING_DIR / 'dnn_feature_maps_extraction')
EEG_DATA_DIR = _path_from_env(
	'EEG_DATA_DIR', DATASET_ROOT / 'THINGS_EEG' / 'Preprocessed_data_250Hz')
IMAGE_SET_DIR = _path_from_env(
	'IMAGE_SET_DIR', DATASET_ROOT / 'THINGS_EEG' / 'images_set')
PRETRAIN_WEIGHTS_DIR = _path_from_env(
	'PRETRAIN_WEIGHTS_DIR', ENCODING_DIR / 'pretrain_weights')

ALEXNET_WEIGHT = os.path.join(PRETRAIN_WEIGHTS_DIR, 'alexnet-owt-7be5be79.pth')
MOCO_WEIGHT = os.path.join(PRETRAIN_WEIGHTS_DIR, 'moco_v1_200ep_pretrain.pth.tar')
CORNET_S_WEIGHT = os.path.join(PRETRAIN_WEIGHTS_DIR, 'cornet_s-1d3f7974.pth')
RESNET50_WEIGHT = os.path.join(PRETRAIN_WEIGHTS_DIR, 'resnet50-0676ba61.pth')
DINO_VITB16_WEIGHT = os.path.join(PRETRAIN_WEIGHTS_DIR, 'dino_vitbase16_pretrain.pth')
DINOV2_VITB14_WEIGHT = os.path.join(PRETRAIN_WEIGHTS_DIR, 'dinov2_vitb14_pretrain.pth')
OPENCLIP_VITB32_WEIGHT = os.path.join(PRETRAIN_WEIGHTS_DIR, 'openclip_vit_b_32_openai.pt')
VIT_B_32_HF_DIR = os.path.join(
	PRETRAIN_WEIGHTS_DIR, 'huggingface', 'google-vit-base-patch32-224-in21k')
VIT_B_16_HF_DIR = os.path.join(
	PRETRAIN_WEIGHTS_DIR, 'huggingface', 'google-vit-base-patch16-224-in21k')

# Fallback only for legacy EEG files that do not store a valid ``times`` vector.
EEG_EPOCH_END_MS = 1000.0

# Brain region channel groups (10-20 system, THINGS-EEG 63-channel montage)
BRAIN_REGIONS = {
	'occipital': ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8'],
	'parietal': ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
	'occipital_parietal': [
		'O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
		'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
	],
	'central': ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
	'frontal': ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
		'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8'],
	'temporal': ['FT9', 'FT7', 'FT8', 'FT10', 'T7', 'T8',
		'TP9', 'TP7', 'TP8', 'TP10'],
	'centro_parietal': ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],
}


def str2bool(value):
	if isinstance(value, bool):
		return value
	value = str(value).lower()
	if value in ('true', '1', 'yes', 'y'):
		return True
	if value in ('false', '0', 'no', 'n'):
		return False
	raise argparse.ArgumentTypeError(f'Cannot interpret "{value}" as bool')


def load_npy_dict(path):
	"""Load a dict saved either as np.save(dict) or as an npz-like mapping."""
	data = np.load(path, allow_pickle=True)
	if isinstance(data, np.ndarray):
		if data.shape == ():
			return data.item()
		raise ValueError(f'Expected a saved dict in {path}, got array shape {data.shape}')
	return data


def normalize_eeg_times_seconds(times, n_timepoints=None):
	"""Return a 1D EEG time axis in seconds from saved preprocessing metadata."""
	times = np.asarray(times, dtype=float).reshape(-1)
	if n_timepoints is not None and len(times) != int(n_timepoints):
		raise ValueError(
			f'EEG times length {len(times)} does not match data timepoints {n_timepoints}')
	if len(times) == 0:
		raise ValueError('EEG times vector is empty')
	# Preprocessing stores seconds. If a future file stores milliseconds, normalize.
	if np.nanmax(np.abs(times)) > 10:
		times = times / 1000.0
	return times


def eeg_times_seconds(n_timepoints):
	"""Fallback per-sample times in seconds for legacy files without times metadata."""
	if n_timepoints < 1:
		raise ValueError('n_timepoints must be >= 1')
	return np.linspace(0.0, EEG_EPOCH_END_MS / 1000.0, int(n_timepoints))


def eeg_times_ms(n_timepoints):
	"""Return per-sample times in milliseconds (0--1000 ms)."""
	return eeg_times_seconds(n_timepoints) * 1000.0


def time_window_indices_from_times(times_seconds, start_ms, end_ms):
	"""Indices into an EEG time axis for an inclusive millisecond window."""
	times_ms = normalize_eeg_times_seconds(times_seconds) * 1000.0
	mask = (times_ms >= start_ms) & (times_ms <= end_ms)
	indices = np.where(mask)[0]
	if indices.size == 0:
		raise ValueError(
			f'Empty time window [{start_ms}, {end_ms}] ms for {len(times_ms)} samples '
			f'(axis {times_ms[0]:.1f}--{times_ms[-1]:.1f} ms)')
	return indices, times_ms[indices]


def time_window_indices(n_timepoints, start_ms, end_ms):
	"""Fallback indices into the legacy 0--1000 ms EEG time axis."""
	return time_window_indices_from_times(eeg_times_seconds(n_timepoints), start_ms, end_ms)


def nearest_time_index(times_seconds, target_ms):
	"""Index of the EEG sample nearest to a millisecond target."""
	times_ms = normalize_eeg_times_seconds(times_seconds) * 1000.0
	return int(np.argmin(np.abs(times_ms - target_ms)))


def resolve_brain_region_channels(region_names):
	"""Resolve one or more brain region names into a deduplicated channel list."""
	if isinstance(region_names, str):
		region_names = [r.strip() for r in region_names.split(',') if r.strip()]

	channels = []
	seen = set()
	for region in region_names:
		if region not in BRAIN_REGIONS:
			available = ', '.join(sorted(BRAIN_REGIONS))
			raise ValueError(
				f'Unknown brain region "{region}". Available: {available}')
		for ch in BRAIN_REGIONS[region]:
			if ch not in seen:
				channels.append(ch)
				seen.add(ch)
	return channels, region_names


def get_experiment_subdir(args):
	"""Build a subdirectory tag encoding the experiment settings."""
	regions_tag = '+'.join(getattr(args, 'brain_regions_list', ['occipital_parietal']))
	return os.path.join(
		f'avg_repetitions-{args.avg_repetitions}',
		f'brain_regions-{regions_tag}',
		f'use_pca-{args.use_pca}',
		f'n_components-{args.n_components:05d}',
		f'time_mode-{args.time_mode}',
		f'time_point_ms-{int(args.time_point_ms):04d}',
	)


def get_pca_output_dir(project_dir, dnn, pretrained, layers):
	return os.path.join(
		project_dir, f'dnn_feature_maps_{dnn}', 'pca_feature_maps',
		dnn, f'pretrained-{pretrained}', f'layers-{layers}')


def get_results_dir(args):
	return os.path.join(
		args.project_dir, 'results', f'sub-{args.sub:02d}',
		'synthetic_eeg_data', 'encoding-linearizing',
		f'subjects-{args.subjects}', f'dnn-{args.dnn}',
		f'pretrained-{args.pretrained}', f'layers-{args.layers}',
		get_experiment_subdir(args),
	)


def get_evaluation_summary_path(args):
	return os.path.join(get_results_dir(args), 'evaluation', 'evaluation_summary.csv')


def print_shell_config():
	values = {
		'SCRIPT_DIR': str(SCRIPT_DIR),
		'PROJECT_DIR': PROJECT_DIR,
		'PCA_SCRIPT_DIR': PCA_SCRIPT_DIR,
		'EEG_DATA_DIR': EEG_DATA_DIR,
		'IMAGE_SET_DIR': IMAGE_SET_DIR,
		'PRETRAIN_WEIGHTS_DIR': PRETRAIN_WEIGHTS_DIR,
	}
	for key, value in values.items():
		print(f'export {key}={shlex.quote(str(value))}')


def _build_path_args(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('command', choices=['shell', 'pca-dir', 'eval-summary'])
	parser.add_argument('--sub', default=1, type=int)
	parser.add_argument('--subjects', default='within', type=str)
	parser.add_argument('--dnn', default='alexnet', type=str)
	parser.add_argument('--pretrained', default=True, type=str2bool)
	parser.add_argument('--layers', default='all', type=str)
	parser.add_argument('--avg_repetitions', default=True, type=str2bool)
	parser.add_argument('--brain_regions', default='occipital_parietal', type=str)
	parser.add_argument('--use_pca', default=True, type=str2bool)
	parser.add_argument('--n_components', default=1000, type=int)
	parser.add_argument('--time_mode', default='all', choices=['all', 'single'])
	parser.add_argument('--time_point_ms', default=120, type=float)
	parser.add_argument('--project_dir', default=PROJECT_DIR, type=str)
	args = parser.parse_args(argv)
	args.brain_regions_list = resolve_brain_region_channels(args.brain_regions)[1]
	return args


def main(argv=None):
	args = _build_path_args(argv)
	if args.command == 'shell':
		print_shell_config()
	elif args.command == 'pca-dir':
		print(get_pca_output_dir(args.project_dir, args.dnn, args.pretrained, args.layers))
	elif args.command == 'eval-summary':
		print(get_evaluation_summary_path(args))


if __name__ == '__main__':
	main()
