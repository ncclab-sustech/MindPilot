<div align="center">

<h2 style="border-bottom: 1px solid lightgray;">BrainFLORA: Uncovering Brain Concept Representation via Multimodal Neural Embeddings</h2>

<!-- Badges and Links Section -->
<div style="display: flex; align-items: center; justify-content: center;">

</div>

<br/>

</div>

BrainFLORA: Uncovering Brain Concept Representation via Multimodal Neural Embeddings


<!--  -->
<img src="fig-overview_00.png" alt="fig-genexample" style="max-width: 80%; height: auto;"/>  

A comparative overview of visual decoding paradigms.


<img src="fig-framework_00.png" alt="Framework" style="max-width: 70%; height: auto;"/>

Overall Architecture of BrainFLORA.


<!-- ## Environment setup -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Environment setup</h2>

Run ``setup.sh`` to quickly create a conda environment that contains the packages necessary to run our scripts; activate the environment with conda activate FLORA.


```
. setup.sh
```

You can also create a new conda environment and install the required dependencies by running
```
conda env create -f environment.yml
conda activate FLORA
```

<!-- ## Prepare for Dataset -->

To download the raw data，you can follow：
Dataset | Download path| Dataset | Download path
:---: | :---:|:---: | :---:
THINGS-EEG1 |  [Download](https://openneuro.org/datasets/ds003825/versions/1.1.0) | THINGS-EEG2 | [Download](https://osf.io/3jk45/)
THINGS-MEG |  [Download](https://openneuro.org/datasets/ds004212/versions/2.0.0)| THINGS-fMRI  |  [Download](https://openneuro.org/datasets/ds004192/versions/1.0.7)
THINGS-Images |  [Download](https://osf.io/rdxy2)

<!-- We will release the processed data (such as THINGS-EEG1, THINGS-EEG2, THINGS-MEG, THINGS-fMRI) on [Huggingface], which can be directly used for training.
 -->


<!-- ## Quick training and test  -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Quick training and test</h2>


#### 1.Visual Retrieval
We provide the script to train the modality encoders for ``joint subject training`` Please modify your data set path and run:
```
cd Retrieval/
python retrieval_joint_train_medformer.py --logger True --gpu cuda:0  --output_dir ./outputs/contrast
```

Additionally, replicating the results of other modalities (e.g. MEG, fMRI) by run
```
cd Retrieval/
python retrieval_joint_train_MEG_rerank_medformer.py --logger True --gpu cuda:0  --output_dir ./outputs/contrast
```
We provide the script to evaluation the models:
```
cd eval/
FLORA_inference.ipynb
```

#### 2.Visual Reconstruction
We provide quick training and inference scripts for ``high level and low level pipeline`` of visual reconstruction. Please modify your data set path and run zero-shot on test dataset:
```
# Train and get multimodal neural embeddings aligned with clip embedding:
python train_unified_encoder_highlevel_diffprior.py --modalities ['eeg', 'meg', 'fmri'] --gpu cuda:0  --output_dir ./outputs/contrast
```

```
# Reconstruct images by assigning modalities and subjects:
python FLORA_inference_reconst.py
```
#### 3.Visual Captioning

We provide scripts for caption generation.
```
# step 1: train feature adapter
python train_unified_encoder_highlevel_diffprior.py --modalities ['eeg', 'meg', 'fmri'] --gpu cuda:0  --output_dir ./outputs/contrast

# step 2: get caption from prior latent
FLORA_inference_caption.ipynb

```

