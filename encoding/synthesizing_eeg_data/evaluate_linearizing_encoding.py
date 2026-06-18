"""Evaluate synthetic EEG from linearizing encoding against biological test EEG."""

import argparse
import os

import numpy as np
from scipy.stats import pearsonr as corr
from sklearn.metrics import explained_variance_score
from sklearn.utils import resample

from encoding_config import (
	EEG_DATA_DIR,
	PROJECT_DIR,
	get_results_dir,
	load_npy_dict,
	normalize_eeg_times_seconds,
	resolve_brain_region_channels,
	str2bool,
	time_window_indices_from_times,
)


def load_biological_test_eeg(sub, selected_channels, time_start_ms, time_end_ms):
	"""Load test EEG and slice an inclusive millisecond window on saved EEG times."""
	eeg_file = os.path.join(
		EEG_DATA_DIR, f'sub-{sub:02d}', 'preprocessed_eeg_test.npy')
	data = load_npy_dict(eeg_file)

	ch_names = list(data['ch_names'])
	selected_indices = [
		ch_names.index(ch) for ch in selected_channels if ch in ch_names]
	ch_names = [ch_names[i] for i in selected_indices]

	y_test = data['preprocessed_eeg_data'][:, :, selected_indices, :]
	n_timepoints = y_test.shape[-1]
	times = normalize_eeg_times_seconds(data['times'], n_timepoints)
	t_idx, times_ms = time_window_indices_from_times(
		times, time_start_ms, time_end_ms)
	y_test = y_test[:, :, :, t_idx]
	return y_test, ch_names, times_ms, times


def align_synthetic_to_bio(synthetic_data, synthetic_times, time_start_ms, time_end_ms):
	"""Crop synthetic EEG to the same saved-time window as biological data."""
	t_idx, _ = time_window_indices_from_times(
		synthetic_times, time_start_ms, time_end_ms)
	return {
		layer: layer_data[:, :, t_idx]
		for layer, layer_data in synthetic_data.items()
	}


def compute_correlation_analysis(y_test_pred, y_test, times_ms, n_iter=10, seed=20200220):
	"""Split-half correlation with low/up noise ceiling (matches eeg_encoding correlation.py).

	Main result: synthetic vs Half 1 average (across 200 test conditions).
	noise_ceiling_low: Half 2 avg vs Half 1 avg.
	noise_ceiling_up: All repetitions avg vs Half 1 avg.
	"""
	np.random.seed(seed)
	correlation = {}
	explained_variance = {}
	corr_timeseries = {}
	noise_timeseries_low = {}
	noise_timeseries_up = {}

	# Average across all repetitions for the noise ceiling upper bound
	bio_avg_all = np.mean(y_test, axis=1)

	for layer in y_test_pred.keys():
		correlation[layer] = np.zeros(
			(n_iter, y_test.shape[2], y_test.shape[3]))
		explained_variance[layer] = np.zeros_like(correlation[layer])
		noise_ceiling_low = np.zeros_like(correlation[layer])
		noise_ceiling_up = np.zeros_like(correlation[layer])

		for i in range(n_iter):
			shuffle_idx = resample(
				np.arange(0, y_test.shape[1]), replace=False,
				n_samples=int(y_test.shape[1] / 2))
			bio_half_1 = np.mean(np.delete(y_test, shuffle_idx, 1), 1)
			bio_half_2 = np.mean(y_test[:, shuffle_idx, :, :], 1)

			for t in range(y_test.shape[3]):
				for c in range(y_test.shape[2]):
					correlation[layer][i, c, t] = corr(
						y_test_pred[layer][:, c, t], bio_half_1[:, c, t])[0]
					explained_variance[layer][i, c, t] = explained_variance_score(
						bio_half_1[:, c, t], y_test_pred[layer][:, c, t])
					noise_ceiling_low[i, c, t] = corr(
						bio_half_2[:, c, t], bio_half_1[:, c, t])[0]
					noise_ceiling_up[i, c, t] = corr(
						bio_avg_all[:, c, t], bio_half_1[:, c, t])[0]

		corr_timeseries[layer] = np.mean(correlation[layer], axis=(0, 1))
		noise_timeseries_low[layer] = np.mean(noise_ceiling_low, axis=(0, 1))
		noise_timeseries_up[layer] = np.mean(noise_ceiling_up, axis=(0, 1))

	layer_summary = {}
	for layer in y_test_pred.keys():
		ts = corr_timeseries[layer]
		nc_low_ts = noise_timeseries_low[layer]
		nc_up_ts = noise_timeseries_up[layer]
		peak_idx = int(np.argmax(ts))
		layer_summary[layer] = {
			'mean_correlation': float(np.mean(ts)),
			'peak_correlation': float(ts[peak_idx]),
			'peak_time_ms': float(times_ms[peak_idx]),
			'mean_explained_variance': float(np.mean(explained_variance[layer])),
			'noise_ceiling_low_mean': float(np.mean(nc_low_ts)),
			'noise_ceiling_low_peak': float(np.max(nc_low_ts)),
			'noise_ceiling_up_mean': float(np.mean(nc_up_ts)),
			'noise_ceiling_up_peak': float(np.max(nc_up_ts)),
		}

	noise_ceiling_low = float(np.mean([
		layer_summary[l]['noise_ceiling_low_mean'] for l in layer_summary]))
	noise_ceiling_up = float(np.mean([
		layer_summary[l]['noise_ceiling_up_mean'] for l in layer_summary]))
	return (
		layer_summary, noise_ceiling_low, noise_ceiling_up,
		corr_timeseries, noise_timeseries_low, noise_timeseries_up,
	)


def save_evaluation_results(
		args, layer_summary, noise_ceiling_low, noise_ceiling_up,
		corr_timeseries, noise_timeseries_low, noise_timeseries_up,
		ch_names, times_ms, full_times_ms, time_start_ms, time_end_ms,
		n_timepoints_full):
	results_dir = get_results_dir(args)
	eval_dir = os.path.join(results_dir, 'evaluation')
	os.makedirs(eval_dir, exist_ok=True)

	peak_info = {
		layer: {
			'peak_correlation': stats['peak_correlation'],
			'peak_time_ms': stats['peak_time_ms'],
		}
		for layer, stats in layer_summary.items()
	}

	results = {
		'sub': args.sub,
		'dnn': args.dnn,
		'brain_regions': args.brain_regions,
		'avg_repetitions': args.avg_repetitions,
		'use_pca': args.use_pca,
		'n_components': args.n_components,
		'time_mode': args.time_mode,
		'time_point_ms': args.time_point_ms,
		'eeg_epoch_ms': [float(full_times_ms[0]), float(full_times_ms[-1])],
		'n_timepoints_full': n_timepoints_full,
		'eval_time_window_ms': [time_start_ms, time_end_ms],
		'eval_times_ms': times_ms,
		'channels': ch_names,
		'layer_summary': layer_summary,
		'peak_info': peak_info,
		'corr_timeseries': corr_timeseries,
		'noise_timeseries_low': noise_timeseries_low,
		'noise_timeseries_up': noise_timeseries_up,
		'noise_ceiling_low': noise_ceiling_low,
		'noise_ceiling_up': noise_ceiling_up,
	}
	np.save(os.path.join(eval_dir, 'evaluation_results.npy'), results)

	csv_path = os.path.join(eval_dir, 'evaluation_summary.csv')
	with open(csv_path, 'w') as f:
		f.write(
			'Subject,Model,Layer,Eval_Start_ms,Eval_End_ms,Mean_Correlation,'
			'Peak_Correlation,Peak_Time_ms,Mean_Explained_Variance,'
			'Noise_Ceiling_Low,Noise_Ceiling_Up\n')
		for layer, stats in layer_summary.items():
			f.write(
				f'{args.sub},{args.dnn},{layer},'
				f'{time_start_ms:.0f},{time_end_ms:.0f},'
				f'{stats["mean_correlation"]:.6f},'
				f'{stats["peak_correlation"]:.6f},'
				f'{stats["peak_time_ms"]:.1f},'
				f'{stats["mean_explained_variance"]:.6f},'
				f'{stats["noise_ceiling_low_mean"]:.6f},'
				f'{stats["noise_ceiling_up_mean"]:.6f}\n')
	return results, csv_path


def print_evaluation_summary(args, results, csv_path):
	win = results['eval_time_window_ms']
	print('\n>>> Evaluation summary <<<')
	print(f'Subject: {args.sub:02d}  DNN: {args.dnn}')
	print(f'  EEG axis (full)     : {results["eeg_epoch_ms"][0]:.0f}--'
	      f'{results["eeg_epoch_ms"][1]:.0f} ms, '
	      f'{results["n_timepoints_full"]} samples')
	print(f'  Analysis window     : {win[0]:.0f}--{win[1]:.0f} ms')
	for layer, stats in results['layer_summary'].items():
		print(f'  Layer {layer}:')
		print(f'    mean correlation (window) = {stats["mean_correlation"]:.4f}')
		print(f'    peak correlation          = {stats["peak_correlation"]:.4f} '
		      f'@ {stats["peak_time_ms"]:.0f} ms')
		print(f'    mean explained var        = {stats["mean_explained_variance"]:.4f}')
	print(f'  noise ceiling low (mean)      = {results["noise_ceiling_low"]:.4f}')
	print(f'  noise ceiling up (mean)       = {results["noise_ceiling_up"]:.4f}')
	print(f'  saved to: {csv_path}')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--sub', default=1, type=int)
	parser.add_argument('--dnn', default='alexnet', type=str)
	parser.add_argument('--subjects', default='within', type=str)
	parser.add_argument('--pretrained', default=True, type=str2bool)
	parser.add_argument('--layers', default='all', type=str)
	parser.add_argument('--avg_repetitions', default=True, type=str2bool)
	parser.add_argument('--brain_regions', default='occipital_parietal', type=str)
	parser.add_argument('--use_pca', default=True, type=str2bool)
	parser.add_argument('--n_components', default=1000, type=int)
	parser.add_argument('--time_mode', default='all', choices=['all', 'single'])
	parser.add_argument('--time_point_ms', default=120, type=float)
	parser.add_argument('--project_dir', default=PROJECT_DIR, type=str)
	parser.add_argument('--n_iter', default=10, type=int)
	parser.add_argument('--eval_time_start_ms', default=60, type=float,
		help='Start of analysis window on the saved EEG time axis (default: 60)')
	parser.add_argument('--eval_time_end_ms', default=500, type=float,
		help='End of analysis window on the saved EEG time axis (default: 500)')
	args = parser.parse_args()
	args.brain_regions_list = resolve_brain_region_channels(args.brain_regions)[1]

	selected_channels, _ = resolve_brain_region_channels(args.brain_regions)
	results_dir = get_results_dir(args)
	synthetic_file = os.path.join(results_dir, 'synthetic_eeg_test.npy')
	if not os.path.exists(synthetic_file):
		raise FileNotFoundError(f'Synthetic test EEG not found: {synthetic_file}')

	synthetic = np.load(synthetic_file, allow_pickle=True).item()
	layer_name = next(iter(synthetic['synthetic_data']))
	n_timepoints_full = synthetic['synthetic_data'][layer_name].shape[2]
	synthetic_times = normalize_eeg_times_seconds(
		synthetic['times'], n_timepoints_full)

	y_bio, ch_names, times_ms, bio_times = load_biological_test_eeg(
		args.sub, selected_channels,
		args.eval_time_start_ms, args.eval_time_end_ms)
	if not np.allclose(synthetic_times, bio_times, atol=1e-9):
		raise ValueError(
			'Synthetic EEG times do not match biological EEG times: '
			f'synthetic {synthetic_times[0]:.3f}--{synthetic_times[-1]:.3f}s '
			f'({len(synthetic_times)} samples), biological '
			f'{bio_times[0]:.3f}--{bio_times[-1]:.3f}s ({len(bio_times)} samples)')
	y_pred = align_synthetic_to_bio(
		synthetic['synthetic_data'], synthetic_times,
		args.eval_time_start_ms, args.eval_time_end_ms)

	(layer_summary, noise_ceiling_low, noise_ceiling_up,
	 corr_ts, noise_ts_low, noise_ts_up) = compute_correlation_analysis(
		y_pred, y_bio, times_ms, args.n_iter)
	results, csv_path = save_evaluation_results(
		args, layer_summary, noise_ceiling_low, noise_ceiling_up,
		corr_ts, noise_ts_low, noise_ts_up,
		ch_names, times_ms, bio_times * 1000.0,
		args.eval_time_start_ms, args.eval_time_end_ms,
		n_timepoints_full)
	print_evaluation_summary(args, results, csv_path)


if __name__ == '__main__':
	main()
