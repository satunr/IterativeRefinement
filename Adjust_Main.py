import networkx as nx
import random
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import kendalltau
from copy import deepcopy
from Func import *

# This code simulates an iterative update of a directed GENIE-style disease network
# using synthetic patient trajectories. It first initializes a random directed graph
# with uniform edge weights and constructs a row-normalized transition matrix
# (Relation) consistent with existing edges.

# It then simulates multiple patient sequences as random walks on this transition
# matrix and builds a subnetwork H whose edges are weighted by how frequently
# they appear across patients. Edges in the original graph G that appear in H
# have their weights increased proportionally (controlled by delta), and edges not
# in H are decreased proportionally to preserve the total sum of edge weights.

# It performs a conservation-based re-weighting of the network driven
# by observed patient path frequencies.


def density(G):
    n = len(G)
    return len(G.edges()) / (n * (n - 1))


def sparsify(G0, keep_density=0.00025):
    n = len(G0)
    W = {(u, v): G0[u][v]['weight'] for (u, v) in G0.edges()}
    L = sorted(W.items(), key=lambda x: x[1])

    remove_edge_cnt = int(len(G0.edges()) - keep_density * (n * (n - 1)))
    remove_egde_list = [(u, v) for ((u, v), _) in L]
    print (f'We are removing {remove_edge_cnt} edges out of {len(G0.edges())}.')

    G0.remove_edges_from(remove_egde_list[:remove_edge_cnt])

    return G0


def find_sum(G):
    # This code returns the sum of edges weight in any input network G
    return sum([G[u][v]['weight'] for (u, v) in G.edges()])


# Parameters
how_many_patients_ = 50

# Disease network
length_of_sequence = 25
delta, decay = 0.3, 0.9999
# Default: delta, decay = 0.05, 0.9985

# Ground truth network
'''
G0 = nx.erdos_renyi_graph(n=n, p=0.05, directed=True)
for (u, v) in G0.edges():
    G0[u][v]['weight'] = 0.1
print (G0)
'''

'''
[A, Diseases, _, VIM3] = pickle.load(open('/Users/sr0215/Python/Clinical/Bayes/Refinement/VIM3.p', 'rb'))
VIM3 = deepcopy(VIM3 / np.max(VIM3))

G0 = create_gml(VIM3, Diseases)
n = len(G0)
'''

'''
# G0 = nx.read_gml('GENIE_trimmed.gml')
# print (G0)
G0 = nx.read_gml('Cond_trimmed.gml')
print (G0)

# G0 = sparsify(G0)
# Gcc = sorted(nx.connected_components(G0.to_undirected()), key=len, reverse=True)
# G0 = G0.subgraph(Gcc[0])

d = density(G0)
print (G0)

L = [G0[u][v]['weight'] for (u, v) in G0.edges()]
max_weight = np.max(L)
med_weight = np.median(L)
for (u, v) in G0.edges():
    G0[u][v]['weight'] = G0[u][v]['weight'] / max_weight

Diseases = list(G0.nodes())
All_pairs = [(u, v) for u in Diseases for v in Diseases if u != v]
n = len(G0)

# Initial GENIE network
# G = nx.erdos_renyi_graph(n=n, p=0.05, directed=True)
G = nx.DiGraph()
G.add_nodes_from(Diseases)

for u in Diseases:
    for v in Diseases:
        if u == v:
            continue
        if random.uniform(0, 1) < min(d, 0.0001):
            G.add_edge(u, v, weight=med_weight)
print (f"The ground truth GENIE network has {len(G)} nodes and {len(G.edges())} edges.")

rep, Repeat = 0, 5000
Log, Log_corr = [], []
Runtime_o, Runtime_a = {}, {}
while rep < Repeat:
    # print (f'We are at step {rep}.')
    t0 = time.time()

    edge_freq = defaultdict(float)
    for _ in range(how_many_patients_):
        S = np.random.choice(Diseases, size=5).tolist()
        Seq = spread_reach_chains(G0, S, 1000)
        # print (Seq)
        # continue

        for seq in Seq:
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                edge_freq[(seq[i], seq[i + 1])] += (1.0 / how_many_patients_)

    H = nx.DiGraph()
    for (u, v), freq in edge_freq.items():
        H.add_edge(u, v, frequency=freq)

    weights_sum_old = find_sum(G)
    L = {(u, v): G[u][v]['weight'] for (u, v) in G.edges()
         if (u, v) not in edge_freq}
    sum_L = sum(L.values())

    # Increase weights of edges in G based on presence in H
    for (u, v), freq in edge_freq.items():
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=0.0)
        w = G[u][v]['weight']
        G[u][v]['weight'] = (1 - delta) * w + delta * freq

    Change = find_sum(G) - weights_sum_old
    if sum_L > 0:
        for (u, v), w in L.items():
            new_w = w - (w / sum_L) * Change
            new_w = max(0.0, min(1.0, new_w))

            if new_w == 0:
                G.remove_edge(u, v)
            else:
                G[u][v]['weight'] = new_w

    # sample_edges = np.random.choice(a=[i for i in range(len(All_pairs))], size=10000).tolist()
    # sample_edges = [All_pairs[i] for i in sample_edges]
    if rep % 25 == 0:
        X = [G0[u][v]['weight'] if (u, v) in G0.edges() else 0
                              for u in sorted(G0.nodes()) for v in sorted(G0.nodes())]
        Y = [G[u][v]['weight'] if (u, v) in G.edges() else 0
                              for u in sorted(G0.nodes()) for v in sorted(G0.nodes())]

        # X = [G0[u][v]['weight'] for (u, v) in G0.edges()]
        # Y = [G[u][v]['weight'] if (u, v) in G.edges() else 0 for (u, v) in G0.edges()]

        corr, p = kendalltau(X, Y)

        PCC = np.corrcoef(X, Y)[0, 1]
        Log.append([rep, corr, p])
        Log_corr.append([rep, PCC])
        print(rep, corr, PCC, delta)

        max_neg_logp = np.max([Log[j][2] for j in range(len(Log))])

        Runtime_a[rep] = time.time() - t0
    else:
        Runtime_o[rep] = time.time() - t0
    rep = rep + 1
    delta = delta * decay
    # print (f'It took {Runtime[-1]} seconds for this step.')

    # # Plot runtime
    # plt.clf()
    # plt.plot([t for t in sorted(Runtime_o.keys())],
    #              [np.mean([Runtime_o[ti] for ti in sorted(Runtime_o.keys()) if ti <= t])
    #               for t in sorted(Runtime_o.keys())])
    # plt.errorbar([t for t in sorted(Runtime_o.keys())],
    #              [np.mean([Runtime_o[ti] for ti in sorted(Runtime_o.keys()) if ti <= t])
    #               for t in sorted(Runtime_o.keys())],
    #              yerr=[np.std([Runtime_o[ti] for ti in sorted(Runtime_o.keys()) if ti <= t])
    #                    for t in sorted(Runtime_o.keys())], alpha=0.01)
    # 
    # plt.xlabel('Number of batches', fontsize=14)
    # plt.ylabel('Runtime in seconds', fontsize=14)
    # plt.tight_layout()
    # plt.savefig('Runtime.png', dpi=150)
    # # plt.show()

# pickle.dump([Log, Log_corr, Runtime_a, Runtime_o], open('Adjust.p', 'wb'))
pickle.dump([Log, Log_corr, Runtime_a, Runtime_o], open('Adjust_cond.p', 'wb'))
exit(1)
'''
'''
# [Log, Log_corr, Runtime_a, Runtime_o] = pickle.load(open('Adjust.p', 'rb'))

print (Log[-1][1], np.std([Log[i][1] for i in range(len(Log))]))
print (Log_corr[-1][1], np.std([Log_corr[i][1] for i in range(len(Log_corr))]))

# 0.01, 0.9985
# 0.7773301552180226 0.1612627026110262
# 0.6306771299805837 0.13869773971633517

# 0.1, 0.9985
# 0.7765384847913884 0.15904840819810634
# 0.6351758295639678 0.16801117708885638

# 0.05, 0.9985
# 0.7774056889456797 0.15980873301085013
# 0.6768009061227018 0.15444291771724392

# 0.05, 0.95
# 0.7760623372916241 0.16064289425599143
# 0.07910206901056392 0.005543757683036768

# 0.05, 0.975
# 0.7776194292425964 0.16102012980307331
# 0.16794153411394808 0.014070711411055203

'''
# Visualize
[Log, Log_corr, Runtime_a, Runtime_o] = pickle.load(open('Adjust_cond.p', 'rb'))
print (f'Runtime is {np.mean(list(Runtime_o.values()))} +- {np.std(list(Runtime_o.values()))}')
# Runtime is 1.0823779117067656 +- 1.0248641376996164

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.errorbar([Log[i][0] for i in range(len(Log))],
         [Log[i][1] for i in range(len(Log))],
             yerr=[np.std([Log[j][1] for j in range(i)]) if i > 1 else 0 for i in range(len(Log))],
             color='red', alpha=0.1)
ax1.plot([Log[i][0] for i in range(len(Log))],
         [Log[i][1] for i in range(len(Log))], color='red')

ax2.plot([Log_corr[i][0] for i in range(len(Log))],
         [Log_corr[i][1] for i in range(len(Log))], 'g-')
ax2.errorbar([Log_corr[i][0] for i in range(len(Log))],
         [Log_corr[i][1] for i in range(len(Log))],
             yerr=[np.std([Log_corr[j][1] for j in range(i)]) if i > 1 else 0 for i in range(len(Log))],
             color='green', alpha=0.1)

ax1.set_xlabel('Number of batches')
ax1.set_ylabel("Kendall's Tau", color='r')
ax2.set_ylabel("Pearson correlation coefficient", color='g')

# plt.title(f'Pearson correlation coefficient is {PCC}')
plt.tight_layout()
plt.savefig('/Users/sr0215/Python/Clinical/Bayes/Refinement/Adjust_cond.png', dpi=100)
plt.show()
