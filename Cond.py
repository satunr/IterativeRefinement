import networkx as nx
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
from sklearn.preprocessing import normalize

from copy import deepcopy
from Func import *
np.set_printoptions(suppress=True, precision=2)


def cluster_genie(VIM3, gene_names, K=17):
    # symmetrize (important)
    W = (VIM3 + VIM3.T) / 2.0

    # convert to distance (higher weight = closer)
    # D = 1.0 - W / np.max(W)
    D = np.exp(-W)
    D_condensed = squareform(D, checks=False)

    # hierarchical clustering
    linkage = sch.linkage(D_condensed, method='average')

    # dendrogram
    # plt.figure(figsize=(12, 6))
    sch.dendrogram(linkage, labels=gene_names, leaf_rotation=90)
    plt.title("GENIE-based Dendrogram")
    plt.savefig('Dendro.png', dpi=300)
    # plt.show()

    # cut into K clusters
    labels = sch.fcluster(linkage, K, criterion='maxclust')

    return labels


def read_diagnosis(fname):
    E = pd.read_csv(fname)

    DID = {}
    Diseases = []
    for i in range(len(E)):
        # if i % 10000 == 0:
        #     print (f'Progress is {i / len(E)}')

        did = E.iloc[i]['ICD9_CODE']
        try:
            did = did[:3]
        except:
            # print (i, did)
            continue

        if did not in Diseases:
            Diseases.append(did)

        sid = E.iloc[i]['SUBJECT_ID']
        if did not in DID.keys():
            DID[did] = []
        DID[did].append(sid)

    n = len(Diseases)
    print (len(Diseases))

    D = np.zeros((n, n))
    for i in range(n):
        if i % 10 == 0:
            print (f'Progress is {i / n}')

        for j in range(n):
            if i == j:
                continue

            D[i, j] = (float(len([sid for sid in DID[Diseases[i]] if sid in DID[Diseases[j]]]))
                       / len(DID[Diseases[i]]))

    return D, Diseases


def cond_prob(A):
    n = A.shape[1]
    C = np.zeros((n, n))

    for i in range(len(A)):
        l = deepcopy(A[i, :]).tolist()
        indices = [j for j in range(len(l)) if l[j] == 1]

        for j in range(len(indices) - 1):
            for k in range(j + 1, len(indices)):
                C[indices[j], indices[k]] += 1
                C[indices[k], indices[j]] += 1

    return C


def concordance(VIM3, Diseases, D):
    labels = cluster_genie(VIM3, Diseases)

    ICD9_categories = list(D.values())
    Mapping = {ICD9_categories[i]: i for i in range(len(ICD9_categories))}
    print (Mapping)

    C = [0, 0]
    for i in range(len(labels) - 1):
        di = Diseases[i][:3]
        if 'E' in di or 'V' in di:
            continue

        category_i = None
        for k in sorted(D.keys()):
            if int (di) <= k:
                category_i = Mapping[D[k]]
                break

        for j in range(i + 1, len(labels)):
            dj = Diseases[j][:3]
            if 'E' in dj or 'V' in dj:
                continue

            category_j = None
            for k in sorted(D.keys()):
                if int(dj) <= k:
                    category_j = Mapping[D[k]]
                    break

            C[1] += 1

            if labels[i] == labels[j] and category_i == category_j:
                C[0] += 1
            elif labels[i] != labels[j] and category_i != category_j:
                C[0] += 1

    return C


'''
D, Diseases = read_diagnosis('DIAGNOSES_ICD.csv')
print (D[:5, :5])

pickle.dump([D, Diseases], open('Cond.p', 'wb'))
'''

[A, Diseases, _, VIM3] = pickle.load(open('/Users/sr0215/Python/Clinical/Bayes/Refinement/VIM3.p', 'rb'))
# print(A[:15, :10])

# G = create_gml(VIM3, Diseases)
# pickle.dump(G, open('GENIE.p', 'wb'))
#
# nx.write_gml(G, 'GENIE.gml')
# exit(1)
#
G = pickle.load(open('GENIE.p', 'rb'))
print(G)

# Diseases = list(G.nodes())
C = cond_prob(A)

# # Generate conditional network graph
# # C = deepcopy(normalize(C, axis=1, norm='l1'))
# I = nx.from_numpy_array(C, create_using=nx.DiGraph)
# print(I)
# nx.write_gml(I, 'Cond.gml')
# exit(1)

# Find conditional probability
X, Y = [], []
for i in range(len(Diseases)):
    for j in range(len(Diseases)):
        if i == j:
            continue
        X.append(G[Diseases[i]][Diseases[j]]['weight'])

        if np.sum(C[i, :]) > 0:
            Y.append(C[i, j] / np.sum(C[i, :]))
        else:
            Y.append(0)

Y = np.array(Y)


# Plot conditional probability
ranges = np.linspace(0, max(X), 5).tolist()
# print (ranges)

Dic = {}
for i in range(len(X)):
    for j in range(len(ranges) - 1):
        if ranges[j] < X[i] <= ranges[j + 1]:
            if ranges[j] not in Dic.keys():
                Dic[ranges[j]] = []
            Dic[ranges[j]].append(Y[i])
            break
# print ({key: np.mean(Dic[key]) for key in sorted(Dic.keys())})
pickle.dump(Dic, open('Cond_dic.p', 'wb'))

Dic = pickle.load(open('Cond_dic.p', 'rb'))
keys = sorted(Dic.keys())
plt.bar(
    range(len(keys)),
    [np.mean(Dic[k]) for k in keys],
    yerr=[np.std(Dic[k]) for k in keys],
    edgecolor='black',
    capsize=5
)
plt.xticks(range(len(keys)), [round(key, 2) for key in keys])
plt.xlabel('GENIE weights', fontsize=15)
plt.ylabel('Conditional probability', fontsize=15)
plt.tight_layout()
plt.savefig('/Users/sr0215/Python/Clinical/Bayes/Refinement/Cond.png', dpi=150)
plt.show()

