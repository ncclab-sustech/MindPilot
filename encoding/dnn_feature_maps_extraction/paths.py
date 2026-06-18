"""Compatibility exports for MindPilot path configuration."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SYNTHESIZING_DIR = SCRIPT_DIR.parent / 'synthesizing_eeg_data'
if str(SYNTHESIZING_DIR) not in sys.path:
	sys.path.insert(0, str(SYNTHESIZING_DIR))

from encoding_config import (  # noqa: E402
	ALEXNET_WEIGHT,
	CORNET_S_WEIGHT,
	DINO_VITB16_WEIGHT,
	DINOV2_VITB14_WEIGHT,
	EEG_DATA_DIR,
	IMAGE_SET_DIR,
	MOCO_WEIGHT,
	OPENCLIP_VITB32_WEIGHT,
	PRETRAIN_WEIGHTS_DIR,
	PROJECT_DIR,
	RESNET50_WEIGHT,
	str2bool,
	VIT_B_16_HF_DIR,
	VIT_B_32_HF_DIR,
)

SCRIPT_DIR = str(SCRIPT_DIR)
