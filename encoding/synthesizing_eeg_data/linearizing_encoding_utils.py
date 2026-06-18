import os

import numpy as np

from encoding_config import (
	EEG_DATA_DIR,
	get_experiment_subdir,
	get_results_dir,
	load_npy_dict,
	nearest_time_index,
	normalize_eeg_times_seconds,
	resolve_brain_region_channels,
)


def _load_full_feature_maps(args, split):
	"""Load and flatten full (non-PCA) feature maps for one image split."""
	base_dir = f'dnn_feature_maps_{args.dnn}'
	fmaps_dir = os.path.join(
		args.project_dir, base_dir, 'full_feature_maps', args.dnn,
		f'pretrained-{args.pretrained}', f'{split}_images')
	fmaps_list = sorted(os.listdir(fmaps_dir))

	fmaps_out = {}
	for fmaps_name in fmaps_list:
		fmaps_data = np.load(
			os.path.join(fmaps_dir, fmaps_name), allow_pickle=True).item()
		all_layers = list(fmaps_data.keys())
		if args.layers == 'all':
			feats = np.concatenate(
				[np.reshape(fmaps_data[layer], -1) for layer in all_layers])
			key = 'all_layers'
			if key not in fmaps_out:
				fmaps_out[key] = []
			fmaps_out[key].append(feats)
		else:
			for layer in all_layers:
				if layer not in fmaps_out:
					fmaps_out[layer] = []
				fmaps_out[layer].append(np.reshape(fmaps_data[layer], -1))

	for key in fmaps_out:
		fmaps_out[key] = np.asarray(fmaps_out[key])
	return fmaps_out


def load_dnn_data(args):
	"""Load DNN feature maps for training and test images."""
	if args.use_pca:
		if args.layers == 'all':
			data_dir = os.path.join(
				f'dnn_feature_maps_{args.dnn}', 'pca_feature_maps',
				args.dnn, f'pretrained-{args.pretrained}', 'layers-all')
		else:
			data_dir = os.path.join(
				f'dnn_feature_maps_{args.dnn}', 'pca_feature_maps',
				args.dnn, f'pretrained-{args.pretrained}', 'layers-single')

		training_file = 'pca_feature_maps_training.npy'
		test_file = 'pca_feature_maps_test.npy'
		X_train = np.load(
			os.path.join(args.project_dir, data_dir, training_file),
			allow_pickle=True).item()
		X_test = np.load(
			os.path.join(args.project_dir, data_dir, test_file),
			allow_pickle=True).item()
	else:
		print('Loading full (non-PCA) feature maps — this may take a while...')
		X_train = _load_full_feature_maps(args, 'training')
		X_test = _load_full_feature_maps(args, 'test')

	if args.layers == 'appended':
		for l, layer in enumerate(X_train.keys()):
			if l == 0:
				train = X_train[layer]
				test = X_test[layer]
			else:
				train = np.append(train, X_train[layer], 1)
				test = np.append(test, X_test[layer], 1)
		X_train = {'appended_layers': train}
		X_test = {'appended_layers': test}

	if args.use_pca:
		for layer in X_train.keys():
			X_train[layer] = X_train[layer][:, :args.n_components]
			X_test[layer] = X_test[layer][:, :args.n_components]

	return X_train, X_test


def load_eeg_data(args):
	"""Load EEG training data with configurable repetitions and brain regions."""
	selected_channels, _ = resolve_brain_region_channels(args.brain_regions)

	all_sub = args.all_sub
	y_train_within = []
	y_train_between = []
	selected_indices = None
	ch_names = None
	times = None

	for s in all_sub:
		eeg_file = os.path.join(
			EEG_DATA_DIR, f'sub-{int(s):02d}', 'preprocessed_eeg_training.npy')
		data = load_npy_dict(eeg_file)

		if selected_indices is None:
			all_ch_names = list(data['ch_names'])
			selected_indices = [
				all_ch_names.index(ch) for ch in selected_channels
				if ch in all_ch_names]
			ch_names = [all_ch_names[i] for i in selected_indices]
			times = normalize_eeg_times_seconds(
				data['times'], data['preprocessed_eeg_data'].shape[-1])

		eeg_raw = data['preprocessed_eeg_data']
		if args.avg_repetitions:
			eeg_data = np.mean(eeg_raw, 1)[:, selected_indices, :]
		else:
			n_images, n_rep, n_ch, n_times = eeg_raw.shape
			eeg_data = eeg_raw.reshape(n_images * n_rep, n_ch, n_times)
			eeg_data = eeg_data[:, selected_indices, :]

		if s == args.sub:
			y_train_within.append(eeg_data)
		else:
			y_train_between.append(eeg_data)

		del data

	if args.subjects == 'within':
		y_train = np.asarray(y_train_within[0])
	elif args.subjects == 'between':
		y_train = np.mean(np.asarray(y_train_between), 0)

	n_timepoints = y_train.shape[-1]
	times = normalize_eeg_times_seconds(times, n_timepoints)

	if args.time_mode == 'single':
		time_idx = nearest_time_index(times, args.time_point_ms)
		y_train = y_train[:, :, time_idx]
		times = np.asarray([times[time_idx]])

	return y_train, ch_names, times


def expand_features_for_repetitions(X, n_repetitions):
	"""Repeat feature rows to match non-averaged EEG repetitions."""
	expanded = {}
	for layer, feats in X.items():
		expanded[layer] = np.repeat(feats, n_repetitions, axis=0)
	return expanded


def perform_regression(args, ch_names, times, X_train, X_test, y_train):
	"""Train linear regression and save synthetic EEG."""
	from ols import OLS_pytorch

	if y_train.ndim == 2:
		eeg_shape = (y_train.shape[0], y_train.shape[1], 1)
		y_flat = y_train
	else:
		eeg_shape = y_train.shape
		y_flat = np.reshape(y_train, (y_train.shape[0], -1))

	synt_train = {}
	synt_test = {}
	betas = None

	for layer in X_train.keys():
		reg = OLS_pytorch(use_gpu=False)
		betas = reg.fit(X_train[layer], y_flat.T)
		betas = np.reshape(
			np.squeeze(np.asarray(betas)), (eeg_shape[1], eeg_shape[2], -1))

		synt_train[layer] = np.reshape(
			reg.predict(X_train[layer]),
			(X_train[layer].shape[0], eeg_shape[1], eeg_shape[2]))
		synt_test[layer] = np.reshape(
			reg.predict(X_test[layer]),
			(X_test[layer].shape[0], eeg_shape[1], eeg_shape[2]))

	save_dir = get_results_dir(args)
	os.makedirs(save_dir, exist_ok=True)

	for split_name, synt_data in [('training', synt_train), ('test', synt_test)]:
		data_dict = {
			'synthetic_data': synt_data,
			'ch_names': ch_names,
			'times': times,
			'betas': betas,
			'experiment_settings': {
				'avg_repetitions': args.avg_repetitions,
				'brain_regions': args.brain_regions,
				'use_pca': args.use_pca,
				'n_components': args.n_components,
				'time_mode': args.time_mode,
				'time_point_ms': args.time_point_ms,
			},
		}
		np.save(
			os.path.join(save_dir, f'synthetic_eeg_{split_name}.npy'), data_dict)

	print(f'Saved synthetic EEG to: {save_dir}')
