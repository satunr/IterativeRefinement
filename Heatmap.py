import pickle
import scipy.cluster.hierarchy as sch
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from scipy.spatial.distance import squareform
from copy import deepcopy
from Func import *
np.set_printoptions(suppress=True, precision=2)


def cluster_genie2(VIM3, gene_names, K=17):
    # Symmetrize
    W = (VIM3 + VIM3.T) / 2.0

    # Convert to distance
    D = np.exp(-W)
    np.fill_diagonal(D, 0)

    D_condensed = squareform(D, checks=False)
    print(D_condensed.shape)

    # Hierarchical clustering
    linkage = sch.linkage(D_condensed, method='average')

    # Dendrogram (optional)
    sch.dendrogram(linkage, labels=gene_names, leaf_rotation=90)

    # Raw cluster labels
    labels = sch.fcluster(linkage, K, criterion='maxclust')

    labels = np.array(labels)
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])

    return linkage, labels


def heatmap_icd(Aggregate, num_clusters=17, topk=3):
    plt.clf()

    mat = np.zeros((num_clusters, topk))
    annot = [["" for _ in range(topk)] for _ in range(num_clusters)]

    for i in range(num_clusters):
        if i not in Aggregate or len(Aggregate[i]) == 0:
            continue

        Each = Counter(Aggregate[i]).most_common()
        S = sum([each[1] for each in Each])

        top = [(each[0], each[1] / S) for each in Each[:topk]]
        for j, (label, val) in enumerate(top):
            mat[i, j] = val
            labels = str(label.replace('Diseases of the ', ''))
            labels = str(labels.replace('Disorders', ''))
            annot[i][j] = labels  # ICD9 description inside cell

    # plt.figure(figsize=(8, 6))
    sns.heatmap(mat, annot=annot, fmt="", cmap="YlGnBu",
                cbar=True, linewidths=0.5, annot_kws={"fontsize": 7,
                                                      "fontweight": "bold"})

    plt.xlabel("Top ICD9 categories")
    plt.ylabel("Cluster ID")
    # plt.title("ICD9 Distribution per Cluster (Top 3)")

    plt.tight_layout()
    plt.savefig('Clustering.png', dpi=150)
    plt.show()


def concordance(VIM3, Diseases, D):
    sch, labels = cluster_genie2(VIM3, Diseases)

    ICD9_categories = list(D.values())
    Mapping = {ICD9_categories[i]: i for i in range(len(ICD9_categories))}
    Inv_Mapping = {Mapping[i]: i for i in Mapping.keys()}

    Groups = {}
    C = [0, 0]
    for i in range(len(labels) - 1):
        di = Diseases[i][:3]

        if not di.isdigit():
            continue

        category_i = None
        for k in sorted(D.keys()):
            if int (di) <= k:
                category_i = Mapping[D[k]]
                break

        if i not in Groups.keys():
            Groups[i] = Inv_Mapping[category_i]

        for j in range(i + 1, len(labels)):
            dj = Diseases[j][:3]
            if not dj.isdigit():
                continue

            category_j = None
            for k in sorted(D.keys()):
                if int(dj) <= k:
                    category_j = Mapping[D[k]]
                    break
            if j not in Groups.keys():
                Groups[j] = Inv_Mapping[category_j]

            C[1] += 1

            if labels[i] == labels[j] and category_i == category_j:
                C[0] += 1
            elif labels[i] != labels[j] and category_i != category_j:
                C[0] += 1
    # print (Groups)

    # Find the distribution of ICD9 in each of the 17 GENIE3 clusters
    Aggregate = {i: [] for i in range(max(labels) + 1)}
    for i in range(len(Diseases)):
        if not i in Groups.keys():
            continue
        Aggregate[labels[i]].append(Groups[i])

    for i in Aggregate.keys():
        if len(Aggregate[i]) == 0:
            continue
        Each = Counter(Aggregate[i]).most_common()

        S = sum([each[1] for each in Each])
        print (i, [(each[0], each[1] / S) for each in Each[:3]])
        print ('\n')

    heatmap_icd(Aggregate)
    return C


def prune_clusters(VIM3, Diseases, K=17, min_cluster_size=10):
    VIM3 = VIM3.copy()
    Diseases = list(Diseases)
    prev_n = -1

    while True:
        linkage, labels = cluster_genie2(VIM3, Diseases, K=K)
        # print(labels)

        # Count cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        print (cluster_sizes)

        # Find clusters to keep
        keep_clusters = {c for c, sz in cluster_sizes.items() if sz >= min_cluster_size}

        # Indices to keep
        keep_idx = [i for i, lab in enumerate(labels) if lab in keep_clusters]

        # If nothing changes → convergence
        if len(keep_idx) == prev_n or len(keep_idx) == len(Diseases):
            break

        prev_n = len(keep_idx)

        # Prune
        VIM3 = deepcopy(VIM3[np.ix_(keep_idx, keep_idx)])
        Diseases = deepcopy([Diseases[i] for i in keep_idx])
        print (f'There are {len(Diseases)} left in the world.')

    return VIM3, Diseases, labels


[A, Diseases, _, VIM3] = pickle.load(open('/Users/sr0215/Python/Clinical/Bayes/Refinement/VIM3.p', 'rb'))

# VIM3 = deepcopy(VIM3[:1000, :1000])
# Diseases = deepcopy(Diseases[:1000])

# VIM3 = np.random.rand(1000, 1000)
# Diseases = [i for i in range(1000)]

# Remove isolated diseases nodes to find large clusters
VIM3, Diseases, labels = prune_clusters(VIM3, Diseases)
print(labels)

# Find concordance index and the distribution of the ICD9 codes in each GENIE cluster
D = pickle.load(open('/Users/sr0215/Python/Clinical/Bayes/Refinement/parse.p', 'rb'))
D = {int(key): D[key] for key in D.keys()}
C = concordance(VIM3, Diseases, D)
print (C[0] / C[1])
# 0.5859301043713834

