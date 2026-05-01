# Overall Goals:
# 1. take in gml and create a networkx graph from it.
# 2. create edge weight vs. frequency plot of gml graph.
# 3. find the knee point of that plot.

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
from pathlib import Path

# Allow imports from the repository root when this file is executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Func import *
from Genie import GENIE3
from kneePoint import detect_knee_point, plot_knee


DEFAULT_BIN_SIZE = 0.0001
DEFAULT_KNEE_AGGRESSIVENESS = 0


# step 1: load gml and create networkx graph from it.
def load_ground_truth_gml(path):
    """Load GML and recover from malformed numpy scalar tokens if needed."""
    gml_path = Path(path)
    if not gml_path.is_absolute():
        cwd_candidate = Path.cwd() / gml_path
        repo_candidate = REPO_ROOT / gml_path
        local_candidate = Path(__file__).resolve().parent / gml_path

        if cwd_candidate.exists():
            gml_path = cwd_candidate
        elif repo_candidate.exists():
            gml_path = repo_candidate
        else:
            gml_path = local_candidate

    try:
        return nx.read_gml(gml_path, label='id')
    except nx.NetworkXError:
        raw_text = gml_path.read_text(encoding='utf-8')
        fixed_text = re.sub(r'NP\.FLOAT64\(([^\)]+)\)', r'\1', raw_text)
        if fixed_text == raw_text:
            raise
        # Persist a clean file so future reads work without patching.
        gml_path.write_text(fixed_text, encoding='utf-8')
        return nx.parse_gml(fixed_text.splitlines(), label='id')
    

# step 2: create edge weight vs. frequency plot of gml graph.
def plot_edge_weight_distribution(graph, title="Edge Weight Distribution", bin_size=DEFAULT_BIN_SIZE):
    edge_weights = [data['weight'] for _, _, data in graph.edges(data=True)]
    if not edge_weights:
        raise ValueError("Graph has no edges to plot")

    min_weight = float(np.min(edge_weights))
    max_weight = float(np.max(edge_weights))
    bin_edges = np.arange(min_weight, max_weight + bin_size, bin_size)
    if bin_edges.size < 2:
        bin_edges = np.array([min_weight, max_weight + bin_size])

    counts, bin_edges = np.histogram(edge_weights, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    frequencies = counts / counts.max() if counts.max() > 0 else counts

    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, frequencies, color='blue', linewidth=2)
    plt.title(title)
    plt.xlabel('Edge Weight')
    plt.ylabel('Normalized Frequency')
    plt.xlim(min_weight - 0.05, max_weight)
    plt.ylim(-0.05, 1)
    plt.grid(alpha=0.75)
    plt.savefig(Path(__file__).resolve().parent / "edge_weight_distribution.png")


# step 3: find the knee point of that plot. put the knee point on the plot as well.
# need to extract the x and y values from the plot to feed into the knee point detection function.
def extract_plot_data_for_knee_detection(graph, bin_size=DEFAULT_BIN_SIZE):
    edge_weights = [data['weight'] for _, _, data in graph.edges(data=True)]
    if not edge_weights:
        raise ValueError("Graph has no edges to extract data from")

    min_weight = float(np.min(edge_weights))
    max_weight = float(np.max(edge_weights))
    bin_edges = np.arange(min_weight, max_weight + bin_size, bin_size)
    if bin_edges.size < 2:
        bin_edges = np.array([min_weight, max_weight + bin_size])

    counts, bin_edges = np.histogram(edge_weights, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    frequencies = counts / counts.max() if counts.max() > 0 else counts

    return bin_centers, frequencies


# step 4: calculate the total number of edges past the knee threshold and print that out as well.
def calculate_edges_past_knee(graph, knee_x):
    edge_weights = [data['weight'] for _, _, data in graph.edges(data=True)]
    edges_past_knee = sum(1 for w in edge_weights if w > knee_x)
    return edges_past_knee


# Step 1 and 2
G = load_ground_truth_gml("GENIE.gml")
print (G)

# G = nx.DiGraph()
# for u in G.nodes():
#     for v in G.nodes():
#         G.add_edge(u, v, weight=random.uniform(0, 1))

# plot_edge_weight_distribution(G,title="Edge Weight Distribution",bin_size=DEFAULT_BIN_SIZE,)

# step 3
x_arr, y_arr = extract_plot_data_for_knee_detection(G, bin_size=DEFAULT_BIN_SIZE)
knee_result = detect_knee_point(
    x_arr,
    y_arr,
    aggressiveness=DEFAULT_KNEE_AGGRESSIVENESS,
)
print("Detected mode:", f"{knee_result.direction} {knee_result.curve} curve,")
print("Knee point (x, y):", (f"{knee_result.knee_x:.4f}", f"{knee_result.knee_y:.4f}"))
print("Knee aggressiveness:", DEFAULT_KNEE_AGGRESSIVENESS)
plot_knee(knee_result, title="Knee Point Cutoff")

'''
# step 4
edges_past_knee = calculate_edges_past_knee(G, knee_result.knee_x)
print(f"Total number of edges in the graph: {G.number_of_edges()}")
print(f"Total number of edges past the knee threshold ({knee_result.knee_x:.4f}): {edges_past_knee}")
'''

print(G)
G.remove_edges_from([(u, v) for (u, v) in G.edges() if G[u][v]['weight'] < knee_result.knee_x])
print(G)

Rem = []
Diseases = list(sorted((G.nodes())))
for i in range(len(Diseases) - 1):
    for j in range(i + 1, len(Diseases)):
        u = Diseases[i]
        v = Diseases[j]

        if G.has_edge(u, v) and G.has_edge(v, u):
            if G[u][v]['weight'] >= G[v][u]['weight']:
                Rem.append((v, u))
            else:
                Rem.append((u, v))

print(len(Rem))
G.remove_edges_from(Rem)
print(G)
# nx.write_gml(G, 'GENIE_trimmed.gml')
