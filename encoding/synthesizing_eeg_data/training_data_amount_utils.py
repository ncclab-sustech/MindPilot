"""Shared utilities for training-data amount analyses."""

import os

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score
from sklearn.utils import resample

from encoding_config import (
	BRAIN_REGIONS,
	EEG_DATA_DIR,
	resolve_brain_region_channels,
	time_window_indices_from_times,
)
from ols import OLS_pytorch


def load_dnn_data(args, cond_idx):
	"""Load PCA feature maps and keep the selected image conditions."""
	if args.layers == 'all':
		layer_dir = 'layers-all'
	else:
		layer_dir = 'layers-single'
	data_dir = os.path.join(
		args.project_dir,
		f'dnn_feature_maps_{args.dnn}',
		'pca_feature_maps',
		args.dnn,
		'pretrained-' + str(args.pretrained),
		layer_dir,
	)
	X_train = np.load(
		os.path.join(data_dir, 'pca_feature_maps_training.npy'),
		allow_pickle=True,
	).item()
	X_test = np.load(
		os.path.join(data_dir, 'pca_feature_maps_test.npy'),
		allow_pickle=True,
	).item()

	if args.layers == 'appended':
		train, test = None, None
		for layer in X_train.keys():
			train = X_train[layer] if train is None else np.append(train, X_train[layer], axis=1)
			test = X_test[layer] if test is None else np.append(test, X_test[layer], axis=1)
		X_train = {'appended_layers': train}
		X_test = {'appended_layers': test}

	for layer in X_train.keys():
		X_train[layer] = X_train[layer][cond_idx, :args.n_components]
		X_test[layer] = X_test[layer][:, :args.n_components]
	return X_train, X_test


def _selected_channel_indices(ch_names, brain_regions):
	if brain_regions == 'all':
		return list(range(len(ch_names))), list(ch_names)
	selected_channels, _regions = resolve_brain_region_channels(brain_regions)
	missing = [ch for ch in selected_channels if ch not in ch_names]
	if missing:
		print(f'Warning: missing EEG channels skipped: {missing}')
	indices = [ch_names.index(ch) for ch in selected_channels if ch in ch_names]
	if not indices:
		available = ', '.join(ch_names)
		raise ValueError(f'No requested channels found. Available channels: {available}')
	return indices, selected_channels


def load_eeg_data(args, cond_idx, rep_idx):
	"""Load EEG data for selected image conditions, repetitions, channels, and time window."""
	data_dir = os.path.join(EEG_DATA_DIR, f'sub-{args.sub:02d}')
	train_data = np.load(
		os.path.join(data_dir, 'preprocessed_eeg_training.npy'),
		allow_pickle=True,
	)
	test_data = np.load(
		os.path.join(data_dir, 'preprocessed_eeg_test.npy'),
		allow_pickle=True,
	)

	ch_names = list(train_data['ch_names'])
	channel_idx, _selected_channels = _selected_channel_indices(ch_names, args.brain_regions)
	time_idx, times_ms = time_window_indices_from_times(
		train_data['times'],
		args.time_start_ms,
		args.time_end_ms,
	)

	y_train = train_data['preprocessed_eeg_data'][cond_idx]
	y_train = np.mean(y_train[:, rep_idx], axis=1)
	y_train = y_train[:, channel_idx, :]
	y_train = y_train[:, :, time_idx]

	y_test = test_data['preprocessed_eeg_data']
	y_test = y_test[:, :, channel_idx, :]
	y_test = y_test[:, :, :, time_idx]
	return y_train, y_test, times_ms / 1000.0


def perform_regression(X_train, X_test, y_train):
	"""Fit OLS from DNN features to EEG responses and predict test EEG."""
	eeg_shape = y_train.shape
	y_train = np.reshape(y_train, (y_train.shape[0], -1))
	y_test_pred = {}
	for layer in X_train.keys():
		reg = OLS_pytorch(use_gpu=False)
		reg.fit(X_train[layer], y_train.T)
		y_test_pred[layer] = np.reshape(
			reg.predict(X_test[layer]),
			(-1, eeg_shape[1], eeg_shape[2]),
		)
	return y_test_pred


def _safe_pearson(x, y):
	if np.std(x) == 0 or np.std(y) == 0:
		return np.nan
	return pearsonr(x, y)[0]


def _noise_split_indices(n_reps):
	if n_reps < 2:
		raise ValueError('Noise ceiling requires at least two EEG repetitions.')
	return resample(np.arange(n_reps), replace=False, n_samples=max(1, n_reps // 2))


def correlation_analysis(args, y_test_pred, y_test):
	"""Compute scalar or time-resolved correlation and explained variance."""
	n_noise_iter = getattr(args, 'n_noise_iter', args.n_iter)
	n_channels = y_test.shape[2]
	n_times = y_test.shape[3]
	correlation = {
		layer: np.zeros((n_noise_iter, n_channels, n_times))
		for layer in y_test_pred.keys()
	}
	explained_variance = {
		layer: np.zeros((n_noise_iter, n_channels, n_times))
		for layer in y_test_pred.keys()
	}
	noise_ceiling_low = np.zeros((n_noise_iter, n_channels, n_times))
	noise_ceiling_up = np.zeros((n_noise_iter, n_channels, n_times))
	bio_data_avg_all = np.mean(y_test, axis=1)

	for i in range(n_noise_iter):
		shuffle_idx = _noise_split_indices(y_test.shape[1])
		bio_data_avg_half_1 = np.mean(np.delete(y_test, shuffle_idx, axis=1), axis=1)
		bio_data_avg_half_2 = np.mean(y_test[:, shuffle_idx, :, :], axis=1)

		for t in range(n_times):
			for c in range(n_channels):
				for layer in y_test_pred.keys():
					pred = y_test_pred[layer][:, c, t]
					target = bio_data_avg_half_1[:, c, t]
					correlation[layer][i, c, t] = _safe_pearson(pred, target)
					explained_variance[layer][i, c, t] = explained_variance_score(target, pred)
				noise_ceiling_low[i, c, t] = _safe_pearson(
					bio_data_avg_half_2[:, c, t],
					bio_data_avg_half_1[:, c, t],
				)
				noise_ceiling_up[i, c, t] = _safe_pearson(
					bio_data_avg_all[:, c, t],
					bio_data_avg_half_1[:, c, t],
				)

	if args.analysis == 'timeseries':
		return {
			'correlation': {
				layer: np.nanmean(values, axis=(0, 1))
				for layer, values in correlation.items()
			},
			'explained_variance': {
				layer: np.nanmean(values, axis=(0, 1))
				for layer, values in explained_variance.items()
			},
			'noise_ceiling_low': np.nanmean(noise_ceiling_low, axis=(0, 1)),
			'noise_ceiling_up': np.nanmean(noise_ceiling_up, axis=(0, 1)),
		}

	return {
		'correlation': {
			layer: float(np.nanmean(values))
			for layer, values in correlation.items()
		},
		'explained_variance': {
			layer: float(np.nanmean(values))
			for layer, values in explained_variance.items()
		},
		'noise_ceiling_low': float(np.nanmean(noise_ceiling_low)),
		'noise_ceiling_up': float(np.nanmean(noise_ceiling_up)),
	}


def aggregate_runs(run_results):
	"""Average metrics across outer data-resampling iterations."""
	metric_keys = run_results[0].keys()
	aggregated = {}
	for key in metric_keys:
		value = run_results[0][key]
		if isinstance(value, dict):
			aggregated[key] = {
				layer: np.nanmean([run[key][layer] for run in run_results], axis=0)
				for layer in value.keys()
			}
		else:
			aggregated[key] = np.nanmean([run[key] for run in run_results], axis=0)
	return aggregated


def save_data(args, results, times=None):
	"""Save scalar or timeseries results using the existing directory layout."""
	if args.analysis == 'timeseries':
		save_dir = os.path.join(
			args.project_dir,
			'results',
			f'sub-{args.sub:02d}',
			'training_data_amount_analysis_timeseries',
			'dnn-' + args.dnn,
			'pretrained-' + str(args.pretrained),
			'layers-' + args.layers,
			'n_components-' + format(args.n_components, '05'),
		)
		file_name = (
			'training_data_amount_timeseries_n_img_cond-'
			+ format(args.n_img_cond, '06')
			+ '_n_eeg_rep-'
			+ format(args.n_eeg_rep, '02')
		)
		results_dict = {
			'correlation_timeseries': results['correlation'],
			'explained_variance_timeseries': results['explained_variance'],
			'noise_ceiling_timeseries': results['noise_ceiling_low'],
			'noise_ceiling_low_timeseries': results['noise_ceiling_low'],
			'noise_ceiling_up_timeseries': results['noise_ceiling_up'],
			'times': times,
			'description': 'Time-resolved training-data amount analysis',
		}
	else:
		save_dir = os.path.join(
			args.project_dir,
			'results',
			f'sub-{args.sub:02d}',
			'training_data_amount_analysis',
			'dnn-' + args.dnn,
			'pretrained-' + str(args.pretrained),
			'layers-' + args.layers,
			'n_components-' + format(args.n_components, '05'),
		)
		file_name = (
			'training_data_amount_n_img_cond-'
			+ format(args.n_img_cond, '06')
			+ '_n_eeg_rep-'
			+ format(args.n_eeg_rep, '02')
		)
		results_dict = {
			'correlation': results['correlation'],
			'explained_variance': results['explained_variance'],
			'noise_ceiling_low': results['noise_ceiling_low'],
			'noise_ceiling_up': results['noise_ceiling_up'],
		}

	os.makedirs(save_dir, exist_ok=True)
	np.save(os.path.join(save_dir, file_name), results_dict)
	return os.path.join(save_dir, file_name + '.npy')
