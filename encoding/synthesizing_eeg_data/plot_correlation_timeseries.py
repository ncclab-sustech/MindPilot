"""Plot correlation timeseries results to match paper's Figure 4.

This script loads the timeseries correlation results and creates plots similar
to the paper's Figure 4, showing correlation coefficients as a function of time.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from encoding_config import PROJECT_DIR, str2bool

def plot_correlation_timeseries(args):
    """Plot correlation timeseries for a single subject."""

    # Load the timeseries results
    save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
        format(args.sub,'02'), 'training_data_amount_analysis_timeseries', 'dnn-'+args.dnn,
        'pretrained-'+str(args.pretrained), 'layers-'+args.layers,
        'n_components-'+format(args.n_components,'05'))
    file_name = 'training_data_amount_timeseries_n_img_cond-'+\
        format(args.n_img_cond,'06')+'_n_eeg_rep-'+format(args.n_eeg_rep,'02')+'.npy'

    try:
        results = np.load(os.path.join(save_dir, file_name), allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Results file not found: {os.path.join(save_dir, file_name)}")
        print("Please run training_data_amount.py --analysis timeseries first!")
        return

    correlation = results['correlation_timeseries']
    noise_ceiling = results['noise_ceiling_timeseries']
    times = results['times']

    # Convert times to milliseconds for better readability
    times_ms = times * 1000

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot correlation timeseries for each layer
    colors = plt.cm.Set1(np.linspace(0, 1, len(correlation.keys())))
    for i, (layer, corr_data) in enumerate(correlation.items()):
        ax.plot(times_ms, corr_data, label=f'{layer}', color=colors[i], linewidth=2)

    # Plot noise ceiling
    ax.plot(times_ms, noise_ceiling, label='Noise Ceiling',
            color='gray', linestyle='--', linewidth=2, alpha=0.7)

    # Add zero line
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)

    # Add stimulus onset line
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Stimulus Onset')

    # Formatting
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel("Pearson's correlation coefficient")
    ax.set_title(f'Correlation Timeseries - Subject {args.sub:02d} - {args.dnn}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set reasonable y-axis limits
    all_data = np.concatenate([list(correlation.values()), [noise_ceiling]])
    y_min = max(-0.1, np.min(all_data) - 0.05)
    y_max = min(1.0, np.max(all_data) + 0.05)
    ax.set_ylim(y_min, y_max)

    # Add statistics text
    stats_text = []
    for layer, corr_data in correlation.items():
        peak_corr = np.max(corr_data)
        peak_time = times_ms[np.argmax(corr_data)]
        stats_text.append(f'{layer}: peak={peak_corr:.3f} at {peak_time:.0f}ms')

    ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    plot_save_dir = os.path.join(save_dir, 'plots')
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)

    plot_filename = f'correlation_timeseries_sub{args.sub:02d}_{args.dnn}.png'
    plt.savefig(os.path.join(plot_save_dir, plot_filename), dpi=300, bbox_inches='tight')

    print(f"Plot saved to: {os.path.join(plot_save_dir, plot_filename)}")
    plt.show()

    return fig, ax

def plot_multiple_subjects(args, subjects=[1,2,3,4,5,6,7,8,9,10]):
    """Plot correlation timeseries averaged across multiple subjects."""

    all_correlations = {}
    all_noise_ceilings = []
    times = None

    # Load data for all subjects
    valid_subjects = []
    for sub in subjects:
        save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
            format(sub,'02'), 'training_data_amount_analysis_timeseries', 'dnn-'+args.dnn,
            'pretrained-'+str(args.pretrained), 'layers-'+args.layers,
            'n_components-'+format(args.n_components,'05'))
        file_name = 'training_data_amount_timeseries_n_img_cond-'+\
            format(args.n_img_cond,'06')+'_n_eeg_rep-'+format(args.n_eeg_rep,'02')+'.npy'

        try:
            results = np.load(os.path.join(save_dir, file_name), allow_pickle=True).item()
            correlation = results['correlation_timeseries']
            noise_ceiling = results['noise_ceiling_timeseries']
            if times is None:
                times = results['times']

            # Store data
            for layer, corr_data in correlation.items():
                if layer not in all_correlations:
                    all_correlations[layer] = []
                all_correlations[layer].append(corr_data)

            all_noise_ceilings.append(noise_ceiling)
            valid_subjects.append(sub)

        except FileNotFoundError:
            print(f"Results not found for subject {sub}, skipping...")
            continue

    if len(valid_subjects) == 0:
        print("No valid results found! Please run training_data_amount.py --analysis timeseries first!")
        return

    print(f"Plotting results for {len(valid_subjects)} subjects: {valid_subjects}")

    # Convert to arrays and compute statistics
    for layer in all_correlations.keys():
        all_correlations[layer] = np.array(all_correlations[layer])
    all_noise_ceilings = np.array(all_noise_ceilings)

    # Convert times to milliseconds
    times_ms = times * 1000

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot correlation timeseries for each layer
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_correlations.keys())))
    for i, (layer, corr_data) in enumerate(all_correlations.items()):
        mean_corr = np.mean(corr_data, axis=0)
        sem_corr = stats.sem(corr_data, axis=0)

        ax.plot(times_ms, mean_corr, label=f'{layer}', color=colors[i], linewidth=3)
        ax.fill_between(times_ms, mean_corr - sem_corr, mean_corr + sem_corr,
                       color=colors[i], alpha=0.2)

    # Plot noise ceiling
    mean_noise = np.mean(all_noise_ceilings, axis=0)
    sem_noise = stats.sem(all_noise_ceilings, axis=0)
    ax.plot(times_ms, mean_noise, label='Noise Ceiling',
            color='gray', linestyle='--', linewidth=3, alpha=0.8)
    ax.fill_between(times_ms, mean_noise - sem_noise, mean_noise + sem_noise,
                   color='gray', alpha=0.2)

    # Add zero line
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)

    # Add stimulus onset line
    ax.axvline(x=60, color='black', linestyle='--', alpha=0.7, label='Analysis Window Start')

    # Formatting
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel("Pearson's correlation coefficient")
    ax.set_title(f'Average Correlation Timeseries (n={len(valid_subjects)}) - {args.dnn}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set reasonable y-axis limits
    all_data = np.concatenate([np.concatenate(list(all_correlations.values())),
                              all_noise_ceilings.flatten()])
    y_min = max(-0.1, np.min(all_data) - 0.05)
    y_max = min(1.0, np.max(all_data) + 0.05)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    # Save the plot
    save_dir = os.path.join(args.project_dir, 'results', 'group_analysis',
                           'training_data_amount_analysis_timeseries', 'dnn-'+args.dnn,
                           'pretrained-'+str(args.pretrained), 'layers-'+args.layers,
                           'n_components-'+format(args.n_components,'05'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plot_filename = f'correlation_timeseries_group_{args.dnn}.png'
    plt.savefig(os.path.join(save_dir, plot_filename), dpi=300, bbox_inches='tight')

    print(f"Group plot saved to: {os.path.join(save_dir, plot_filename)}")
    plt.show()

    return fig, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', default=1, type=int, help='Subject number (for single subject plot)')
    parser.add_argument('--dnn', default='alexnet', type=str)
    parser.add_argument('--pretrained', default=True, type=str2bool)
    parser.add_argument('--layers', default='all', type=str)
    parser.add_argument('--n_components', default=1000, type=int)
    parser.add_argument('--n_img_cond', default=4135, type=int)
    parser.add_argument('--n_eeg_rep', default=1, type=int)
    parser.add_argument('--project_dir', default=PROJECT_DIR, type=str)
    parser.add_argument('--plot_type', default='single', choices=['single', 'group'],
                       help='Plot single subject or group average')

    args = parser.parse_args()

    print(f'>>> Plotting correlation timeseries ({args.plot_type}) <<<')

    if args.plot_type == 'single':
        plot_correlation_timeseries(args)
    elif args.plot_type == 'group':
        plot_multiple_subjects(args)
