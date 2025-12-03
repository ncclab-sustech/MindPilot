<div align="center">

<h1>MindPilot</h1>

<h3>Closed-loop Visual Stimulation Optimization for Brain Modulation with EEG-guided Diffusion</h3>

<!-- Badges
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) -->

---

</div>

## 📖 Overview

**MindPilot** is a closed-loop visual stimulation optimization framework for brain modulation. By combining EEG signals with diffusion models, we generate optimized visual stimuli that can effectively modulate brain activity. Both simulation and human EEG experiments demonstrate significant improvements in brain modulation performance.

<!-- ### ✨ Key Features

- 🔄 **Closed-loop Optimization**: Real-time EEG feedback for iterative visual stimuli refinement
- 🎨 **Diffusion-based Generation**: Leveraging state-of-the-art diffusion models for high-quality image generation
- 🧪 **Two Optimization Strategies**: 
  - Interactive Search: Evolutionary search in the latent space
  - Heuristic Generation: Gradient-based optimization with EEG guidance
- 🧠 **Multiple Brain Features**: Support for EEG latent features, PSD features, and custom brain signals
- 📊 **Comprehensive Validation**: Tested on multiple emotion datasets (THINGS-EEG2, ArtPhoto, GAPED, EmoSet)
- 🖥️ **Real-time Deployment**: Client-server architecture for online human experiments

--- -->

## 🎯 Conceptualization

<div align="center">
<img src="fig-conceptualization.png" alt="Conceptualization" width="90%"/>
<p><i>The conceptualization of MindPilot: Closed-loop optimization of visual stimuli using EEG feedback</i></p>
</div>

---

## 🏗️ Architecture

<div align="center">
<img src="fig-framework.png" alt="Framework" width="90%"/>
<p><i>Overall architecture of MindPilot framework</i></p>
</div>
<!-- 
The framework consists of three main components:
1. **EEG Readout Model**: Predicts brain responses from visual features
2. **Optimization Engine**: Searches/generates optimized visual stimuli
3. **Validation System**: Real-time EEG recording and validation -->

---

<!-- ## 📁 Project Structure

```
MindPilot/
├── model/                          # Core models
│   ├── end_to_end.py              # EEG readout model training
│   ├── pseudo_target_model.py     # Target brain response modeling
│   ├── modulation_utils.py        # Brain modulation utilities
│   └── subject_layers/            # Subject-specific neural layers
├── experiments/                    # Experiment scripts
│   ├── exp-interactive_search.ipynb              # Interactive search demo
│   ├── exp-heuristic_generation_*.ipynb         # Heuristic generation demos
│   ├── benchmark_framework_total.py             # Benchmark evaluation
│   └── util.py                                   # Experiment utilities
├── client/                         # Real-time client for human experiments
│   ├── client.py                  # EEG client interface
│   ├── neuracle_api.py            # Neuracle EEG device API
│   └── pygame_utils.py            # Visual stimuli presentation
├── server/                         # Server for online experiments
│   └── improved_experiment.py     # Experiment control server
├── Interactive_search/             # Visualization for interactive search
├── Heuristic_generation/          # Visualization for heuristic generation
└── environment.yml                # Conda environment file
``` -->
<!-- 
---

## 🚀 Getting Started

### Prerequisites

- **OS**: Linux (Ubuntu 18.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (at least 16GB VRAM recommended)
- **CUDA**: Version 11.7+
- **Conda**: Miniconda or Anaconda -->

### Installation

#### Option 1: Quick Setup (Recommended)

```bash
cd MindPilot
chmod +x setup.sh
./setup.sh
conda activate MindPilot
```


#### Option 2: Manual Setup

```bash
conda env create -f environment.yml
conda activate MindPilot
```


---

## 📊 Dataset Preparation

### Download Datasets

Download the required datasets from the following sources:

| Dataset | Description | Download Link |
|:-------:|:------------|:--------------|
| **THINGS-EEG2** | Natural images with EEG responses | [OSF](https://osf.io/3jk45/) |
| **ArtPhoto** | Artistic photographs with emotion ratings | [ImageEmotion](https://www.imageemotion.org) |
| **GAPED** | Geneva Affective Picture Database | [UNIGE](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/) |
| **EmoSet** | Large-scale emotion dataset | [VCC Tech](https://vcc.tech/EmoSet) |


---

## 🎓 Usage

### 1. Train EEG Readout Model

Train a neural network to predict EEG responses from visual features:

```bash
python model/end_to_end.py \
    --dnn alexnet \
    --sub 10 \
    --modeled_time_points all \
    --pretrained False \
    --epochs 50 \
    --lr 1e-5 \
    --weight_decay 0. \
    --batch_size 64 \
    --save_trained_models True \
    --project_dir eeg_encoding/
```


### 2. Interactive Search

Perform evolutionary search in the latent space for optimal stimuli:

```bash
jupyter notebook experiments/exp-interactive_search.ipynb
```


### 3. Heuristic Generation

Generate optimized visual stimuli using gradient-based optimization:

```bash
python experiments/exp-heuristic_generation_with_guidance_anyfeature.py
```

### 4. Benchmark Evaluation

#### Offline Generation Benchmark
```bash
bash experiments/run_benchmark_offline_generation.sh
```

#### Heuristic Generation Benchmark
```bash
bash experiments/run_benchmark_heuristic_generation.sh
```

#### Complete Benchmark Suite
```bash
bash experiments/run_benchmark_total.sh
```

### 5. Real-time Human Experiments

**Server side (runs optimization):**
```bash
python server/improved_experiment.py --port 5000
```

**Client side (presents stimuli and records EEG):**
```bash
python client/client.py --server_ip 192.168.1.100 --port 5000
```

## 📜 License

This project will be released under an open-source license upon paper acceptance.

---

<div align="center">
<p><i>🚧 This is an anonymous repository for peer review. Full code and documentation will be released upon paper acceptance. 🚧</i></p>
</div>

