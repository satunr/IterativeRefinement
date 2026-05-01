import networkx as nx
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt


def dfs_subgraph_(G, n):
    """
    G: directed NetworkX graph
    n: number of nodes to include in subgraph
    """

    if n <= 0:
        return nx.DiGraph()

    # Start from node with highest out-degree
    start = max(G.out_degree, key=lambda x: x[1])[0]

    visited = set()
    stack = [start]

    while stack and len(visited) < n:
        node = stack.pop()

        if node not in visited:
            visited.add(node)

            # Add neighbors (DFS: stack → LIFO)
            neighbors = list(G.successors(node))
            stack.extend(neighbors)

    # If DFS couldn't reach n nodes, fill randomly
    if len(visited) < n:
        remaining = [u for u in G.nodes() if u not in visited]
        visited.update(remaining[:(n - len(visited))])

    # Return induced subgraph
    return G.subgraph(visited).copy()


def color(G, mapping, D):

    H = nx.DiGraph()
    Rem = []
    for u in G.nodes():
        for icd in sorted(D.keys()):
            try:
                if int(mapping[int(u)][:3]) <= icd:
                    H.add_node(mapping[int(u)], weight=icd)
                    # print (icd)
                    break
            except:
                # print (u, mapping[int(u)])
                Rem.append(mapping[int(u)])
                continue

    for (u, v) in G.edges():
        w = G[u][v]['weight']
        H.add_edge(mapping[int(u)], mapping[int(v)], weight=w)

    H.remove_nodes_from(Rem)
    for u in H.nodes():
        if 'weight' not in H.nodes[u].keys():
            print ('Here:', u)

    return H


G = nx.read_gml('GENIE_trimmed.gml')
H = dfs_subgraph_(G, 500)

mapping = pickle.load(open('mapping.p', 'rb'))
print (mapping)
# exit(1)

D = pickle.load(open('/Users/sr0215/Python/Clinical/Bayes/Refinement/parse.p', 'rb'))
D = {int(key): D[key] for key in D.keys()}
print (D)

H = color(H, mapping, D)
# Gcc = sorted(nx.connected_components(H.to_undirected()),
#              key=len, reverse=True)
# H = H.subgraph(Gcc[0])
nx.write_gml(H, 'GENIE_trimmed_colored.gml')


'''
# Accuracy over parameters

# # Varying delta
# Y_KT = [0.7773301552180226, 0.7774056889456797, 0.7765384847913884]
# Err_KT = [0.1612627026110262, 0.15980873301085013, 0.15904840819810634]
#
# Y_PCC = [0.6306771299805837, 0.6768009061227018, 0.6351758295639678]
# Err_PCC = [0.13869773971633517, 0.15444291771724392, 0.16801117708885638]
# X = [0.01, 0.05, 0.10]

# Varying decay
Y_KT = [0.7760623372916241, 0.7776194292425964, 0.7774056889456797]
Err_KT = [0.16064289425599143, 0.16102012980307331, 0.15980873301085013]

Y_PCC = [0.07910206901056392, 0.16794153411394808, 0.6768009061227018]
Err_PCC = [0.005543757683036768, 0.014070711411055203, 0.15444291771724392]
X = [0.95, 0.975, 0.9985]

x = np.arange(len(X))
width = 0.35
plt.bar(x - width/2, Y_KT, width, yerr=Err_KT, capsize=5,
        label='KT', color='red', alpha=0.7)
plt.bar(x + width/2, Y_PCC, width, yerr=Err_PCC, capsize=5,
        label='PCC', color='green', alpha=0.7)

plt.xticks(x, X)
# plt.ylim([0.45, 1.0])
plt.xlabel('Decay ($\gamma$)', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('Sensitivity_Decay.png', dpi=150)
plt.show()
'''
