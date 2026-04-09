import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import glob
import os
import argparse
import base64
from io import BytesIO


def get_palette(n):
    cmap = cm.get_cmap('tab10' if n <= 10 else 'tab20', n)
    return [cmap(i) for i in range(n)]


def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.yaxis.grid(True, linestyle='-', color='#EEEEEE', linewidth=1.0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=10,
                   colors='#888888', length=0, width=0)
    ax.set_xlabel('Communication round', fontsize=11, color='#999999', labelpad=8)


def plot_client_metrics(algo, is_iid, return_base64=False):
    folder_suffix = "_results_iid" if is_iid else "_results"
    data_type     = "IID — Perfectly Balanced" if is_iid else "Non-IID — Highly Skewed"

    plotter_dir    = os.path.dirname(os.path.abspath(__file__))
    results_dir    = os.path.abspath(os.path.join(plotter_dir, '..', 'results'))
    search_pattern = os.path.join(results_dir, f"{algo}{folder_suffix}", "hospital_*_metrics.csv")
    csv_files      = sorted(glob.glob(search_pattern))

    if not csv_files:
        print(f"❌  No CSV files found for '{algo}' in {algo}{folder_suffix}.")
        print(f"    Expected path: results/{algo}{folder_suffix}/hospital_*_metrics.csv")
        return

    print(f"📊  Found {len(csv_files)} client files for {algo.upper()} [{data_type}].")

    colors = get_palette(len(csv_files))

    plt.style.use('default')
    plt.rcParams.update({
        'font.family':      'sans-serif',
        'font.size':        11,
        'figure.facecolor': 'white',
        'axes.facecolor':   'white',
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.07, right=0.97, wspace=0.28)

    fig.suptitle(
        f'Local Hospital Training Performance — {algo.upper()}\n{data_type}',
        fontsize=14, fontweight='normal', color='#333333', y=0.98
    )

    handles = []
    for idx, file in enumerate(csv_files):
        basename = os.path.basename(file)
        try:
            h_id  = basename.split('_')[1]
            label = f"Hospital {h_id}"
        except IndexError:
            label = basename

        color = colors[idx]
        df    = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        line, = ax1.plot(
            df['Round'], df['Accuracy'] * 100,
            color=color, linewidth=2.0, alpha=0.9, solid_capstyle='round'
        )
        ax2.plot(
            df['Round'], df['Loss'],
            color=color, linewidth=2.0, alpha=0.9, solid_capstyle='round'
        )

        final_acc  = df['Accuracy'].iloc[-1] * 100
        final_loss = df['Loss'].iloc[-1]
        final_rnd  = df['Round'].iloc[-1]

        ax1.annotate(
            f'{final_acc:.1f}%',
            xy=(final_rnd, final_acc),
            xytext=(4, 0), textcoords='offset points',
            fontsize=9, color=color, va='center'
        )
        ax2.annotate(
            f'{final_loss:.3f}',
            xy=(final_rnd, final_loss),
            xytext=(4, 0), textcoords='offset points',
            fontsize=9, color=color, va='center'
        )

        handles.append((line, label))

    ax1.set_ylabel('Accuracy (%)', fontsize=11, color='#888888', labelpad=8)
    ax2.set_ylabel('Loss',         fontsize=11, color='#888888', labelpad=8)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    for ax in (ax1, ax2):
        style_axis(ax)

    fig.legend(
        [h[0] for h in handles],
        [h[1] for h in handles],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(csv_files),
        fontsize=10,
        frameon=True,
        facecolor='#F8F9FA',
        edgecolor='#E0E0E0',
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.8,
    )

    if return_base64:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot local hospital metrics for a specific FL algorithm.")
    parser.add_argument('--algo', type=str, default='fedavg',
                        help="The algorithm to plot (e.g., fedavg, wsm_ce_fedavg_nosmote).")
    parser.add_argument('--iid', action='store_true',
                        help="Read from the IID-specific results folder.")
    args = parser.parse_args()

    plot_client_metrics(args.algo, args.iid)