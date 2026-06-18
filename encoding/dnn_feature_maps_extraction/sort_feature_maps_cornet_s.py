"""Compatibility sorter for legacy CORnet-S feature maps.

The current MindPilot extractor saves a full-layer dict directly for each
image, so this script is normally a no-op. If older per-layer files exist in
``*_individual_layers`` directories, this script combines them into the direct
full-layer format used by PCA and linear encoding.

Parameters
----------
pretrained : bool
	If True use a pretrained network, if false a randomly initialized one.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from paths import PROJECT_DIR, str2bool


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=str2bool)
parser.add_argument('--project_dir', default=PROJECT_DIR, type=str)
args = parser.parse_args()

print('>>> Sort feature maps CORnet-S <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load, sort and save the CORnet-S feature maps
# =============================================================================
layers = ['V1', 'V2', 'V4', 'IT', 'decoder']
fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps_cornet_s',
	'full_feature_maps', 'cornet_s', 'pretrained-'+str(args.pretrained))
img_partitions = ['training_images', 'test_images']
# img_partitions = ['training_images', 'test_images', 'ILSVRC2012_img_val',
# 	'ILSVRC2012_img_test_v10102019']
# num_partition_imgs = [16540, 200, 50000, 100000]
num_partition_imgs = [16540, 200]


for p, part in enumerate(img_partitions):
	save_dir = os.path.join(args.project_dir, 'dnn_feature_maps_cornet_s',
		'full_feature_maps', 'cornet_s', 'pretrained-'+str(args.pretrained),
		part)
	os.makedirs(save_dir, exist_ok=True)

	existing_files = [
		f for f in os.listdir(save_dir)
		if f.endswith('.npy') and not f.endswith('_individual_layers')]
	if existing_files:
		sample = np.load(
			os.path.join(save_dir, sorted(existing_files)[0]),
			allow_pickle=True).item()
		if all(layer in sample for layer in layers):
			print(f'{part}: direct full-layer files already exist; skipping')
			continue

	legacy_dir = os.path.join(fmaps_dir, part + '_individual_layers')
	if not os.path.isdir(legacy_dir):
		raise FileNotFoundError(
			f'No direct full-layer files and no legacy directory: {legacy_dir}')

	for i in range(num_partition_imgs[p]):
		model_feats = {}
		for l in layers:
			model_feats[l] = np.asarray(np.load(os.path.join(legacy_dir,
				part+'_layer-'+l+'_'+format(i+1, '07')+
				'.npy'), allow_pickle=True).item()[l])
		file_name = part + '_' + format(i+1, '07')
		np.save(os.path.join(save_dir, file_name), model_feats)
