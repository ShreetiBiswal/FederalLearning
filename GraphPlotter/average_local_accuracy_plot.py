import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import argparse
import base64
from io import BytesIO

PALETTE = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD', '#BA7517']


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


def plot_server_metrics(return_base64=False):
    parser = argparse.ArgumentParser(description="Plot Global Server Metrics.")
    parser.add_argument('--iid', action='store_true', help="Flag to plot IID-specific metrics")
    args = parser.parse_args()

    suffix    = "_iid" if args.iid else ""
    data_type = "IID — Perfectly Balanced" if args.iid else "Non-IID — Highly Skewed"

    plotter_dir = os.path.dirname(os.path.abspath(__file__))
    server_dir  = os.path.abspath(os.path.join(plotter_dir, '..', 'server'))
    csv_file    = os.path.join(server_dir, 'serverAvgMetrics', f'server_avg_metrics{suffix}.csv')

    if not os.path.exists(csv_file):
        print(f"❌  Could not find {csv_file}.")
        return

    print(f"📊  Loading {data_type} data from {csv_file}...")
    df         = pd.read_csv(csv_file)
    algorithms = df['Algorithm'].unique()

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
        f'Federated Learning — Global Model Performance\n{data_type}',
        fontsize=14, fontweight='normal', color='#333333', y=0.98
    )

    handles = []
    for idx, algo in enumerate(algorithms):
        color = PALETTE[idx % len(PALETTE)]
        label = algo.replace('_', ' ').upper()
        data  = df[df['Algorithm'] == algo].sort_values('Round')

        line, = ax1.plot(
            data['Round'], data['Global_Accuracy'] * 100,
            color=color, linewidth=2.0, alpha=0.9, solid_capstyle='round'
        )
        ax2.plot(
            data['Round'], data['Global_Loss'],
            color=color, linewidth=2.0, alpha=0.9, solid_capstyle='round'
        )

        final_acc  = data['Global_Accuracy'].iloc[-1] * 100
        final_loss = data['Global_Loss'].iloc[-1]
        final_rnd  = data['Round'].iloc[-1]

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
        ncol=len(algorithms),
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
    plot_server_metrics()