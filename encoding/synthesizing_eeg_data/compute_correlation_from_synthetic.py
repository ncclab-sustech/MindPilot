"""
Compute correlation results from already synthesized EEG data.

This script reads the pre-computed synthetic EEG data and computes correlation
and explained variance with the biological EEG test data, then saves the results
following the original directory structure.

Parameters
----------
project_dir : str
    Directory of the project folder.
"""

import argparse
import numpy as np
import os
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import pearsonr as corr
from sklearn.metrics import explained_variance_score
from encoding_config import BRAIN_REGIONS, EEG_DATA_DIR, PROJECT_DIR

def load_biological_eeg_test_data(sub):
    """Load the biological EEG test data for a subject.

    Parameters
    ----------
    sub : int
        Subject number.

    Returns
    -------
    y_test : ndarray
        Biological EEG test data.
    """
    selected_channels = BRAIN_REGIONS['occipital_parietal']

    data_dir = os.path.join(EEG_DATA_DIR, f'sub-{sub:02d}')
    test_file = 'preprocessed_eeg_test.npy'
    data = np.load(os.path.join(data_dir, test_file), allow_pickle=True)

    # Get channel indices for selected visual channels
    ch_names = data['ch_names']
    selected_indices = [ch_names.index(ch) for ch in selected_channels if ch in ch_names]

    y_test = data['preprocessed_eeg_data']
    times = np.round(data['times'], 2)

    # Select only visual channels
    y_test = y_test[:, :, selected_indices, :]

    # Select the time points between 60-500ms
    times_start = np.where(times == 0.06)[0][0]
    times_end = np.where(times == 0.51)[0][0]
    y_test = y_test[:, :, :, times_start:times_end]

    return y_test

def compute_correlation_analysis(y_test_pred, y_test, n_iter=10):
    """Compute correlation and explained variance analysis.

    Parameters
    ----------
    y_test_pred : dict
        Predicted test EEG data.
    y_test : ndarray
        Biological test EEG data.
    n_iter : int
        Number of iterations for correlation analysis.

    Returns
    -------
    correlation : dict
        Correlation results.
    explained_variance : dict
        Explained variance results.
    noise_ceiling : float
        Noise ceiling results.
    """
    # Results matrices
    correlation = {}
    explained_variance = {}
    for layer in y_test_pred.keys():
        correlation[layer] = np.zeros((n_iter, y_test.shape[2], y_test.shape[3]))
        explained_variance[layer] = np.zeros((n_iter, y_test.shape[2], y_test.shape[3]))
    noise_ceiling = np.zeros((n_iter, y_test.shape[2], y_test.shape[3]))

    for i in range(n_iter):
        # Random data repetitions index
        shuffle_idx = resample(np.arange(0, y_test.shape[1]), replace=False,
            n_samples=int(y_test.shape[1]/2))
        # Average across one half of the biological data repetitions
        bio_data_avg_half_1 = np.mean(np.delete(y_test, shuffle_idx, 1), 1)
        # Average across the other half of the biological data repetitions for
        # the noise ceiling calculation
        bio_data_avg_half_2 = np.mean(y_test[:,shuffle_idx,:,:], 1)

        # Compute the metrics
        for t in range(y_test.shape[3]):
            for c in range(y_test.shape[2]):
                for layer in y_test_pred.keys():
                    # Correlation
                    correlation[layer][i,c,t] = corr(y_test_pred[layer][:,c,t],
                        bio_data_avg_half_1[:,c,t])[0]
                    # Explained Variance
                    explained_variance[layer][i,c,t] = explained_variance_score(
                        bio_data_avg_half_1[:,c,t], y_test_pred[layer][:,c,t])
                # Noise ceiling
                noise_ceiling[i,c,t] = corr(bio_data_avg_half_2[:,c,t],
                    bio_data_avg_half_1[:,c,t])[0]

    # Average the results across iterations, EEG channels and time points
    for layer in y_test_pred.keys():
        correlation[layer] = np.mean(correlation[layer])
        explained_variance[layer] = np.mean(explained_variance[layer])
    noise_ceiling = np.mean(noise_ceiling)

    return correlation, explained_variance, noise_ceiling

def save_correlation_results(project_dir, sub, dnn, pretrained, layers, n_components,
                           n_img_cond, n_eeg_rep, correlation, explained_variance, noise_ceiling):
    """Save the correlation results following the original directory structure.

    Parameters
    ----------
    project_dir : str
        Project directory.
    sub : int
        Subject number.
    dnn : str
        DNN model name.
    pretrained : bool
        Whether the model is pretrained.
    layers : str
        Layer configuration.
    n_components : int
        Number of PCA components.
    n_img_cond : int
        Number of image conditions.
    n_eeg_rep : int
        Number of EEG repetitions.
    correlation : dict
        Correlation results.
    explained_variance : dict
        Explained variance results.
    noise_ceiling : float
        Noise ceiling results.
    """
    # Store the results into a dictionary
    results_dict = {
        'correlation': correlation,
        'explained_variance': explained_variance,
        'noise_ceiling': noise_ceiling
    }

    # Save directories following the original structure
    save_dir = os.path.join(project_dir, 'results', f'sub-{sub:02d}',
                           'training_data_amount_analysis', f'dnn-{dnn}',
                           f'pretrained-{pretrained}', f'layers-{layers}',
                           f'n_components-{n_components:05d}')
    file_name = f'training_data_amount_n_img_cond-{n_img_cond:06d}_n_eeg_rep-{n_eeg_rep:02d}'

    # Create the directory if not existing and save the data
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, file_name), results_dict)
    print(f"Results saved to: {os.path.join(save_dir, file_name)}.npy")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', default=PROJECT_DIR, type=str)
    parser.add_argument('--n_iter', default=10, type=int)
    args = parser.parse_args()

    print('>>> Computing correlation from synthetic EEG data <<<')
    print(f'\nProject directory: {args.project_dir}')
    print(f'Number of iterations: {args.n_iter}')
    print('Note: Synthetic EEG data corresponds to full training configuration:')
    print('  - n_img_cond: 16540 (all training images)')
    print('  - n_eeg_rep: 4 (all repetitions averaged)')

    # Set random seed for reproducible results
    seed = 20200220
    np.random.seed(seed)

    # Available subjects
    subjects = list(range(1, 11))  # sub-01 to sub-10

    # Available DNNs
    dnns = ['alexnet', 'cornet_s', 'dino_vit_b_16', 'dino2_vit_b_14', 'moco',
            'openclip_vit_b_32', 'resnet50', 'synclr_vit_b_16', 'vit_b_32']

    # Fixed parameters based on the directory structure
    pretrained = True
    layers = 'all'
    n_components = 1000

    # Training data amount parameters: synthetic EEG data corresponds to full training data
    # (16540 image conditions, 4 EEG repetitions averaged)
    training_params = [
        {'n_img_cond': 16540, 'n_eeg_rep': 4}
    ]

    # Process each combination
    for sub in tqdm(subjects, desc="Processing subjects"):
        print(f"\nProcessing subject {sub}")

        # Load biological EEG test data for this subject
        try:
            y_test_bio = load_biological_eeg_test_data(sub)
            print(f"Loaded biological EEG test data shape: {y_test_bio.shape}")
        except Exception as e:
            print(f"Error loading biological data for subject {sub}: {e}")
            continue

        for dnn in tqdm(dnns, desc="Processing DNNs", leave=False):
            # Path to synthetic EEG data
            synthetic_dir = os.path.join(args.project_dir, 'results', f'sub-{sub:02d}',
                                       'synthetic_eeg_data', 'encoding-linearizing', 'subjects-within',
                                       f'dnn-{dnn}', f'pretrained-{pretrained}', f'layers-{layers}',
                                       f'n_components-{n_components:05d}')

            synthetic_test_file = os.path.join(synthetic_dir, 'synthetic_eeg_test.npy')

            # Check if synthetic data exists
            if not os.path.exists(synthetic_test_file):
                print(f"Synthetic data not found: {synthetic_test_file}")
                continue

            try:
                # Load synthetic EEG test data
                y_test_pred_data = np.load(synthetic_test_file, allow_pickle=True).item()

                # Extract the actual synthetic data
                if 'synthetic_data' in y_test_pred_data:
                    synthetic_data = y_test_pred_data['synthetic_data']
                    # Get the time points that match biological data (60-500ms)
                    times = y_test_pred_data['times']
                    times_rounded = np.round(times, 2)
                    times_start = np.where(times_rounded == 0.06)[0][0]
                    times_end = np.where(times_rounded == 0.51)[0][0]

                    # Extract and crop the synthetic data to match biological data time window
                    y_test_pred = {}
                    for layer_name, layer_data in synthetic_data.items():
                        # Crop to 60-500ms time window to match biological data
                        y_test_pred[layer_name] = layer_data[:, :, times_start:times_end]
                else:
                    # Fallback for older format
                    y_test_pred = {'appended_layers': y_test_pred_data}

                print(f"Loaded synthetic EEG data for {dnn}")
                for layer, data in y_test_pred.items():
                    print(f"  Layer {layer}: {data.shape}")

                # Compute correlation analysis
                correlation, explained_variance, noise_ceiling = compute_correlation_analysis(
                    y_test_pred, y_test_bio, args.n_iter)

                # Save results for the training configuration that matches the synthetic data
                params = training_params[0]  # Only one configuration: full training data
                save_correlation_results(
                    args.project_dir, sub, dnn, pretrained, layers, n_components,
                    params['n_img_cond'], params['n_eeg_rep'],
                    correlation, explained_variance, noise_ceiling
                )

                print(f"Completed correlation analysis for subject {sub}, DNN {dnn}")

            except Exception as e:
                print(f"Error processing subject {sub}, DNN {dnn}: {e}")
                continue

    print("\n>>> Correlation computation completed! <<<")

if __name__ == "__main__":
    main()
