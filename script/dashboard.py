import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# At the top of your script
COMPONENTS_DIR = Path(__file__).parent.parent / 'components'


def create_progress_dashboard():
    # Data
    data = {
        'Screening': {'total': 2478, 'completed': 100},
        'Review': {'total': 0, 'completed': 0},
        'Summarization': {'total': 0, 'completed': 0}
    }
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Pipeline Progress Dashboard', fontsize=16)
    
    # Function to create progress bar
    def create_progress_bar(ax, title, total, completed):
        percentage = (completed/total)*100 if total > 0 else 0
        ax.bar([0], [100], color='lightgray', width=0.5)
        ax.bar([0], [percentage], color='blue', width=0.5)
        ax.set_title(f'{title}\n{completed}/{total}')
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.set_ylabel('Percentage')
    
    # Create three progress bars
    create_progress_bar(ax1, 'Screening', data['Screening']['total'], data['Screening']['completed'])
    create_progress_bar(ax2, 'Review', data['Review']['total'], data['Review']['completed'])
    create_progress_bar(ax3, 'Summarization', data['Summarization']['total'], data['Summarization']['completed'])
    
    plt.tight_layout()
    plt.savefig(COMPONENTS_DIR / 'progress-dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate dashboard
create_progress_dashboard()