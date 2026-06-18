"""Analyze how training image/repetition counts affect EEG encoding quality.

Use one entry point for both historical analyses:

```
python training_data_amount.py --analysis scalar
python training_data_amount.py --analysis timeseries
```
"""

import argparse

import numpy as np
from sklearn.utils import resample
from tqdm import tqdm

from encoding_config import PROJECT_DIR, str2bool
from training_data_amount_utils import (
	aggregate_runs,
	correlation_analysis,
	load_dnn_data,
	load_eeg_data,
	perform_regression,
	save_data,
)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--analysis', default='scalar', choices=['scalar', 'timeseries'])
	parser.add_argument('--sub', default=1, type=int)
	parser.add_argument('--dnn', default='alexnet', type=str)
	parser.add_argument('--pretrained', default=True, type=str2bool)
	parser.add_argument('--layers', default='all', choices=['all', 'single', 'appended'])
	parser.add_argument('--n_components', default=1000, type=int)
	parser.add_argument('--n_img_cond', default=16540, type=int)
	parser.add_argument('--n_eeg_rep', default=4, type=int)
	parser.add_argument('--n_iter', default=10, type=int)
	parser.add_argument('--n_noise_iter', default=10, type=int)
	parser.add_argument('--total_img_conditions', default=16540, type=int)
	parser.add_argument('--total_eeg_repetitions', default=4, type=int)
	parser.add_argument('--time_start_ms', default=60.0, type=float)
	parser.add_argument('--time_end_ms', default=500.0, type=float)
	parser.add_argument('--brain_regions', default='occipital_parietal', type=str)
	parser.add_argument('--project_dir', default=PROJECT_DIR, type=str)
	parser.add_argument('--seed', default=20200220, type=int)
	return parser.parse_args()


def print_settings(args):
	title = 'Training data amount analysis'
	if args.analysis == 'timeseries':
		title += ' (timeseries)'
	print(f'>>> {title} <<<')
	print('\nInput arguments:')
	for key, val in vars(args).items():
		print('{:20} {}'.format(key, val))


def validate_args(args):
	if args.n_img_cond > args.total_img_conditions:
		raise ValueError('--n_img_cond cannot exceed --total_img_conditions')
	if args.n_eeg_rep > args.total_eeg_repetitions:
		raise ValueError('--n_eeg_rep cannot exceed --total_eeg_repetitions')


def run(args):
	validate_args(args)
	np.random.seed(args.seed)
	run_results = []
	times = None

	for _iteration in tqdm(range(args.n_iter)):
		cond_idx = np.sort(resample(
			np.arange(args.total_img_conditions),
			replace=False,
			n_samples=args.n_img_cond,
		))
		rep_idx = np.sort(resample(
			np.arange(args.total_eeg_repetitions),
			replace=False,
			n_samples=args.n_eeg_rep,
		))

		X_train, X_test = load_dnn_data(args, cond_idx)
		y_train, y_test, times = load_eeg_data(args, cond_idx, rep_idx)
		y_test_pred = perform_regression(X_train, X_test, y_train)
		run_results.append(correlation_analysis(args, y_test_pred, y_test))

	results = aggregate_runs(run_results)
	output_path = save_data(args, results, times=times)
	print(f'\nSaved results to: {output_path}')

	if args.analysis == 'timeseries':
		print(f'Time range: {times[0]:.3f} - {times[-1]:.3f} seconds')
		for layer, corr_ts in results['correlation'].items():
			peak_idx = int(np.nanargmax(corr_ts))
			print(
				f'{layer}: peak correlation {corr_ts[peak_idx]:.3f} '
				f'at {times[peak_idx]:.3f}s'
			)


if __name__ == '__main__':
	parsed_args = parse_args()
	print_settings(parsed_args)
	run(parsed_args)
