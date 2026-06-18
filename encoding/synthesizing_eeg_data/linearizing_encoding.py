"""Fit a linear regression to predict EEG data using DNN feature maps."""

import argparse

from encoding_config import PROJECT_DIR, resolve_brain_region_channels, str2bool
from linearizing_encoding_utils import (
	expand_features_for_repetitions,
	load_dnn_data,
	load_eeg_data,
	perform_regression,
)


def print_experiment_settings(args):
	selected_channels, region_names = resolve_brain_region_channels(args.brain_regions)
	print('\n' + '=' * 60)
	print('Experiment settings')
	print('=' * 60)
	print(f'  subject(s)         : {args.sub}')
	print(f'  subjects mode      : {args.subjects}')
	print(f'  dnn model          : {args.dnn}')
	print(f'  pretrained         : {args.pretrained}')
	print(f'  layers             : {args.layers}')
	print(f'  avg repetitions    : {args.avg_repetitions}')
	print(f'  brain regions      : {", ".join(region_names)}')
	print(f'  channels ({len(selected_channels)}) : {", ".join(selected_channels)}')
	print(f'  use pca            : {args.use_pca}')
	print(f'  pca n_components   : {args.n_components}')
	print(f'  time mode          : {args.time_mode}')
	if args.time_mode == 'single':
		print(f'  time point (ms)    : {args.time_point_ms}')
	print(f'  project_dir        : {args.project_dir}')
	print('=' * 60 + '\n')


parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--subjects', default='within', type=str)
parser.add_argument('--all_sub', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--pretrained', default=True, type=str2bool)
parser.add_argument('--layers', default='all', type=str)
parser.add_argument('--avg_repetitions', default=True, type=str2bool)
parser.add_argument('--brain_regions', default='occipital_parietal', type=str,
	help='Comma-separated brain region names, e.g. occipital,parietal')
parser.add_argument('--use_pca', default=True, type=str2bool)
parser.add_argument('--n_components', default=1000, type=int)
parser.add_argument('--time_mode', default='all', choices=['all', 'single'],
	help='Regress all time points jointly, or a single time point')
parser.add_argument('--time_point_ms', default=120, type=float,
	help='Target time point in ms when time_mode=single')
parser.add_argument('--project_dir', default=PROJECT_DIR, type=str)
args = parser.parse_args()

_, args.brain_regions_list = resolve_brain_region_channels(args.brain_regions)

print('>>> Training linearizing encoding model <<<')
print_experiment_settings(args)

X_train, X_test = load_dnn_data(args)

if not args.avg_repetitions:
	n_repetitions = 4
	X_train = expand_features_for_repetitions(X_train, n_repetitions)

y_train, ch_names, times = load_eeg_data(args)
perform_regression(args, ch_names, times, X_train, X_test, y_train)

print('>>> Training complete <<<')
