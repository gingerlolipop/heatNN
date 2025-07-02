import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


def create_modified_parallel_workflow():
    # Initialize directed graph
    G = nx.DiGraph()

    # Add nodes with labels
    nodes = {
        'A': 'Literature Search',
        'D1': 'Inclusion/Exclusion',
        'E1': 'Screening Results',
        'C2': 'Full Text Review',
        'E2': 'Notes',
        'C3': 'Data Extraction',
        'Update': 'Dynamic Summary Update',
    }

    # Add nodes to the graph
    for node, label in nodes.items():
        G.add_node(node, label=label)

    # Add edges
    edges = [
        ('A', 'D1'), ('A', 'C2'), ('A', 'C3'),
        ('D1', 'E1'),
        ('E1', 'C2'),  # Screening Results -> Full Text Review
        ('C2', 'E2'),
        ('E2', 'C3'),  # Notes -> Data Extraction
        ('C3', 'Update'),  # Data Extraction -> Dynamic Summary Update
    ]
    G.add_edges_from(edges)

    # Define node positions for horizontal flow
    pos = {
        'A': (0, 2),
        'D1': (2, 3), 'E1': (4, 3),
        'C2': (2, 2), 'E2': (4, 2),
        'C3': (2, 1), 'Update': (4, 1),
    }

    # Extract labels for nodes
    labels = nx.get_node_attributes(G, 'label')

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Draw edges with arrows and colors
    edge_colors = {
        ('A', 'D1'): 'orange', ('D1', 'E1'): 'orange',
        ('E1', 'C2'): 'green', ('C2', 'E2'): 'green',
        ('A', 'C2'): 'green', ('E2', 'C3'): 'blue',
        ('A', 'C3'): 'blue', ('C3', 'Update'): 'purple',
    }

    for edge, color in edge_colors.items():
        nx.draw_networkx_edges(
            G, pos, edgelist=[edge], ax=ax, edge_color=color, arrowstyle='-|>', arrowsize=20, width=2.5
        )

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000, node_color='lightblue', edgecolors='black')
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=10, font_weight='bold')

    # Add dashed horizontal lines to separate pipelines
    for y in [3, 2, 1]:
        ax.hlines(y, xmin=-1, xmax=5, colors='gray', linestyles='dashed', linewidth=0.8)

    # Add pipeline labels as text annotations
    plt.text(-0.5, 2.9, 'Screening Pipeline', fontsize=12, fontweight='bold', ha='left')  # Adjusted downward
    plt.text(-0.5, 2.4, 'Review Pipeline', fontsize=12, fontweight='bold', ha='left')
    plt.text(-0.5, 1.4, 'Summarization Pipeline', fontsize=12, fontweight='bold', ha='left')

    # Add legend
    legend_elements = [
        mpatches.Patch(color='orange', label='Screening Pipeline'),
        mpatches.Patch(color='green', label='Review Pipeline'),
        mpatches.Patch(color='blue', label='Summarization Pipeline'),
        mpatches.Patch(color='purple', label='Dynamic Update'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True)  # Moved legend down

    # Final touches
    plt.title("Modified Parallel Review Pipeline with Dynamic Updates", fontsize=16)
    plt.axis('off')
    plt.show()


# Generate the workflow diagram
create_modified_parallel_workflow()
