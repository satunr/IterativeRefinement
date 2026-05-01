import numpy as np
import networkx as nx
import random
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from copy import deepcopy
from collections import Counter

np.set_printoptions(precision=3, suppress=True)


def drop_leading(seq):

    if len(seq) == 1 and seq == '0':
        return seq

    while True:
        if seq[0] == '0':
            seq = seq[1:]
        else:
            break
    return seq


def edge_time_heatmap(D):
    """
    D: dict of {(source, target, time): frequency}
    Produces a heatmap where rows = edges (source->target), columns = time, values = frequency
    """
    # Extract unique edges and times
    edges = sorted({(u, v) for u, v, t in D.keys()})
    times = sorted({t for _, _, t in D.keys()})

    # Initialize matrix
    mat = pd.DataFrame(0, index=[f"{u}->{v}" for u, v in edges], columns=times)

    # Fill matrix
    for (u, v, t), freq in D.items():
        mat.loc[f"{u}->{v}", t] = freq

    # Plot heatmap
    plt.figure(figsize=(8, len(edges)*0.5 + 2))
    sns.heatmap(mat, annot=True, fmt="d", cmap="YlGnBu")
    plt.xlabel("Time")
    plt.ylabel("Edge", rotation=90)
    plt.title("Edge Frequency Over Time")
    plt.tight_layout()
    plt.show()


def edge_time_heatmap2(j, D):

    count = {}
    All = {}
    for u, v, t in D.keys():
        if (u, v) not in count.keys():
            count[(u, v)] = 1
        else:
            count[(u, v)] += 1

        if u not in All.keys():
            All[u] = 1
        else:
            All[u] += 1
        if v not in All.keys():
            All[v] = 1
        else:
            All[v] += 1

    return All


def try_comb(A, seq):
    options = [seq]

    while len(seq) > 2:

        if seq[:-1] in A.keys():
            options.append(seq[:-1])
        seq = deepcopy(seq[:-1])

    return options


def freq_temp(L, A):
    D = {}
    Timely = {}
    for seq in L:
        for i in range(len(seq) - 1):

            seq[i] = str(seq[i])
            seq[i + 1] = str(seq[i + 1])

            seq[i] = drop_leading(seq[i])
            seq[i + 1] = drop_leading(seq[i + 1])

            if seq[i] == seq[i + 1]:
                continue
            key = (seq[i], seq[i + 1], i)

            if key not in D.keys():
                D[key] = 1
            else:
                D[key] = D[key] + 1

            if i not in Timely.keys():
                Timely[i] = {}
            if (seq[i], seq[i + 1]) not in Timely[i]:
                Timely[i][(seq[i], seq[i + 1])] = 1
            else:
                Timely[i][(seq[i], seq[i + 1])] += 1

    D = {key: D[key] for key in D.keys() if D[key]}

    return D, Timely


def one_hot(seq, n):
    return [1 if i in seq else 0 for i in range(n)]


def read(fname):

    G = nx.read_gml(fname)
    # G = nx.relabel_nodes(G, {u: int(u) for u in G.nodes()})

    diseases = list(sorted(G.nodes()))
    n = len(G)

    '''
    A = np.zeros((n, n))
    Check = []
    for u in diseases:
        for v in diseases:
            if G.has_edge(u, v):
                if (v, u) in Check:
                    print ('ERROR', (u, v))
                Check.append((u, v))

                A[diseases.index(u), diseases.index(v)] \
                    = G[u][v]['weight']
    '''
    return G, diseases, n


def dist(x, y):
    return len([i for i in range(len(x)) if x[i] != y[i]])


def longest_common_contiguous(a, b):
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    max_len = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                max_len = max(max_len, dp[i][j])
            else:
                dp[i][j] = 0

    return np.exp(-max_len)


def common_contiguous_pairs(a, b):
    # Extract all contiguous pairs
    pairs_a = {(a[i], a[i + 1]) for i in range(len(a) - 1)}
    pairs_b = {(b[i], b[i + 1]) for i in range(len(b) - 1)}

    # Intersection
    common = pairs_a & pairs_b

    return len(common)


def dist_matrix(X):
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # D[i, j] = dist(X[i], X[j])
            # D[i, j] = 1 / (longest_common_contiguous(X[i], X[j]) + 1.0)
            D[i, j] = 1 / (common_contiguous_pairs(X[i], X[j]) + 1.0)

    return D


def sample(G, A, diseases, l, H, n):
    Seq = []
    for h in range(H):
        while True:
            # seed = random.choice([diseases[i] for i in range(n)])
            # seed = random.choice(list(G.nodes()))

            N = sorted(G.nodes())
            P = [G.out_degree(u) for u in N]
            P = [p / sum(P) for p in P]

            seed = np.random.choice(N, p=P, size=1).tolist()[0]

            if len(list(G.successors(seed))) > 0:
                break

        '''
        while len(seq) < l:
            curr = seq[-1]

            # P = deepcopy(A[curr, :].tolist())
            P = 0.9 * A[curr, :] + 0.1 * np.eye(n)[curr]

            # print (curr, P)

            seq.append(np.random.choice(a=[i for i in range(n)],
                                        p=P, size=1)[0])
        '''
        seq = weighted_random_walk(G, seed, l)
        while len(seq) < l:
            seq.append(seq[-1])

        Seq.append(seq)

    return np.array(Seq)


def weighted_random_walk(G, start_node, steps):
    current_node = start_node
    path = [current_node]

    while True:
        neighbors = list(G.successors(current_node))

        if not neighbors:
            next_node = current_node
        else:
            # Extract weights for the outgoing edges
            weights = [G[current_node][neighbor]['weight'] for neighbor in neighbors]

            # Select next node based on relative weights
            next_node = random.choices(neighbors, weights=weights, k=1)[0]

        path.append(next_node)
        current_node = next_node

        if len(path) >= steps:
            break
    return path


# Length and number of sequences; number of clusters
l = 5
H = 5000
nC = 5

icd9_dict = {
    '909': 'Late effects of other and unspecified injury',
    '3708': 'Other keratitis',
    '34': 'Other disorders of eye',
    '3709': 'Unspecified keratitis',
    '908': 'Late effects of external causes of injury',
    '1663': 'Malignant neoplasm of other specified sites',
    '59': 'Other disorders of urinary system',
    '2107': 'Benign neoplasm of other digestive organs',
    '6332': 'Ectopic pregnancy, tubal',
    '466': 'Acute bronchitis',
    '325': 'Phlebitis and thrombophlebitis',
    '3085': 'Stress reaction',
    '61': 'Other disorders of female genital organs',
    '4969': 'Chronic airway obstruction, unspecified',
    '193': 'Malignant neoplasm of thyroid gland',
    '4251': 'Hypertrophic cardiomyopathy',
    '484': 'Pneumonia in infectious diseases',
    '5745': 'Cholelithiasis',
    '518': 'Other diseases of lung',
    '1515': 'Malignant neoplasm of stomach',
    '25': 'Diabetes mellitus',
    '3482': 'Encephalopathy',
    '861': 'Injury to lung',
    '4945': 'Bronchiectasis with acute exacerbation',
    '43': 'Other disorders of intestine',
    '155': 'Malignant neoplasm of liver',
    '472': 'Chronic pharyngitis',
    '100': 'Other infectious diseases',
    '2541': 'Disorders of adrenal glands',
    '2252': 'Benign neoplasm of brain',
    '810': 'Fracture of clavicle',
    '4699': 'Upper respiratory condition, unspecified',
    '5152': 'Postinflammatory pulmonary fibrosis',
    '3137': 'Emotional disturbance of childhood',
    '4858': 'Bronchopneumonia',
    '324': 'Intracranial abscess',
    '5586': 'Gastrointestinal inflammation',
    '112': 'Candidiasis',
    '4029': 'Hypertensive heart disease',
    '71': 'Other eye disorders',
    '3753': 'Lacrimal gland disorder',
    '44': 'Other stomach disorders',
    '249': 'Secondary diabetes mellitus',
    '2357': 'Neoplasm of uncertain behavior',
    '706': 'Diseases of sebaceous glands',
    '2083': 'Leukemia',
    '172': 'Malignant melanoma of skin',
    '5283': 'Stomatitis and mucositis',
    '3227': 'Meningitis',
    '1014': 'Other infections',
    '4398': 'Late effects of cerebrovascular disease',
    '494': 'Bronchiectasis',
    '1591': 'Malignant neoplasm of small intestine',
    '1886': 'Malignant neoplasm of bladder',
    '5941': 'Calculus of urinary tract',
    '4729': 'Chronic pharyngitis, unspecified',
    '4153': 'Pulmonary embolism',
    '1644': 'Malignant neoplasm, unspecified site',
    '6187': 'Genital prolapse',
    '948': 'Burns',
    '2191': 'Benign neoplasm of uterus',
    '1466': 'Malignant neoplasm of oropharynx',
    '2614': 'Nutritional deficiency',
    '632': 'Missed abortion',
    '1849': 'Malignant neoplasm of female genital organs',
    '2618': 'Other malnutrition',
    '299': 'Psychoses',
    '3836': 'Chronic otitis media',
    '3023': 'Sexual disorders',
    '364': 'Disorders of iris',
    '4734': 'Chronic sinusitis',
    '2201': 'Benign neoplasm of ovary',
    '6147': 'Pelvic inflammatory disease',
    '6146': 'Inflammatory disease of uterus',
    '3807': 'Otitis externa',
    '4025': 'Hypertensive heart disease with failure',
    '4404': 'Atherosclerosis',
    '78': 'Other bone disorders',
    '2069': 'Leukemia unspecified',
    '1': 'Cholera',
    '1042': 'Viral infection',
    '3979': 'Valvular heart disease',
    '675': 'Breast infection postpartum',
    '4022': 'Hypertensive heart disease',
    '1318': 'Other infections',
    '49': 'Other urinary disorders',
    '158': 'Malignant neoplasm of retroperitoneum',
    '1457': 'Malignant neoplasm of mouth',
    '493': 'Asthma',
    '510': 'Empyema',
    '143': 'Malignant neoplasm of gum',
    '1939': 'Thyroid cancer unspecified',
    '4994': 'Chronic pulmonary disease',
    '4995': 'Chronic pulmonary disease variant',
    '3202': 'Meningitis',
    '91': 'Injury to blood vessels',
    '2547': 'Endocrine disorders',
    '558': 'Noninfectious gastroenteritis',
    '3550': 'Mononeuritis',
    '648': 'Complications of pregnancy',
    '2998': 'Other psychoses',
    '563': 'Other intestinal obstruction',
    '6142': 'Salpingitis and oophoritis',
    '5': 'Other infections',
    '4408': 'Peripheral vascular disease',
    '393': 'Rheumatic fever',
    '6144': 'Pelvic inflammatory disease',
    '1659': 'Malignant neoplasm respiratory',
    '6781': 'Postpartum complication',
    '6555': 'Fetal abnormality affecting mother',
    '6145': 'Pelvic inflammatory disease unspecified',
    '4677': 'Upper respiratory infection',
    '5278': 'Salivary gland disorder',
    '3323': 'Parkinsonism',
    '3324': 'Other degenerative diseases of the basal ganglia',
    '244': 'Hypothyroidism',
    '366': 'Cataract',
    '4148': 'Ischemic heart disease',
    '477': 'Allergic rhinitis',
    '694': 'Dermatitis',
    '121': 'Other parasitic disease',
    '4215': 'Endocarditis',
    '3555': 'Peripheral nerve disorder',
    '1244': 'Helminthiasis',
    '3505': 'Cranial nerve disorder',
    '4146': 'Coronary artery disease',
    '3563': 'Polyneuropathy',
    '5108': 'Pleural disease',
    '1322': 'Other parasitic infection',
    '5002': 'Pneumoconiosis',
    '1541': 'Malignant neoplasm rectum',
    '6232': 'Noninflammatory disorder female genital',
    '225': 'Benign brain neoplasm',
    '5228': 'Periapical abscess',
    '1230': 'Other helminth infection',
    '792': 'Abnormal findings',
    '1166': 'Mycoses',
    '367': 'Refractive error',
    '40': 'Intestinal obstruction',
    '1695': 'Malignant neoplasm',
    '6339': 'Ectopic pregnancy unspecified',
    '974': 'Poisoning',
    '878': 'Open wound',
    '2964': 'Mood disorder',
    '4058': 'Secondary hypertension',
    '2520': 'Parathyroid disorder',
    '31': 'Other respiratory conditions',
    '1255': 'Helminthiasis',
    '1111': 'Candidiasis',
    '1110': 'Candidiasis unspecified',
    '99': 'Other infectious disease',
    '4340': 'Cerebral infarction',
    '268': 'Vitamin deficiency',
    '1686': 'Malignant neoplasm',
    '5030': 'Pulmonary disease',
    '4690': 'Upper respiratory disease',
    '426': 'Conduction disorder',
    '4446': 'Arterial embolism',
    '5951': 'Cystitis',
    '728': 'Muscle disorder',
    '56': 'Urinary condition',
    '1389': 'Late effects infection',
    '4705': 'Upper respiratory infection',
    '809': 'Fracture of skull',
    '6003': 'Benign prostatic hyperplasia',
    '92': 'Contusion',
    '1309': 'Protozoal disease',
    '101': 'Infectious disease',
    '167': 'Malignant neoplasm',
    '65': 'Normal delivery',
    '919': 'Superficial injury',
    '5718': 'Chronic liver disease',
    '1737': 'Skin cancer',
    '633': 'Ectopic pregnancy',
    '1253': 'Helminth infection',
    '698': 'Pruritus',
    '74': 'Cesarean section',
    '1083': 'Viral infection',
    '4002': 'Malignant hypertension',
    '6004': 'Prostate enlargement',
    '6002': 'Prostate disorder',
    '460': 'Acute nasopharyngitis',
    '2974': 'Paranoid states',
    '2455': 'Thyroiditis',
    '5372': 'Gastric disorder',
    '5383': 'Gastroduodenitis',
    '337': 'Autonomic dysfunction',
    '3676': 'Visual disturbance',
    '1896': 'Malignant neoplasm kidney',
    '5384': 'Duodenitis',
    '880': 'Open wound upper limb',
    '3455': 'Epilepsy',
    '7': 'Other infection',
    '5382': 'Gastritis',
    '312': 'Behavior disorder',
    '5272': 'Salivary gland inflammation',
    '405': 'Hypertension',
    '5373': 'Gastroduodenal ulcer',
    '1828': 'Malignant neoplasm uterus',
    '1250': 'Helminthiasis',
    '282': 'Anemia',
    '5371': 'Gastric ulcer',
    '890': 'Open wound lower limb',
    '2363': 'Neoplasm uncertain behavior',
    '1010': 'Infection unspecified',
    '543': 'Appendicitis',
    '5070': 'Aspiration pneumonia',
    '629': 'Infertility',
    '347': 'Sleep disorder',
    '620': 'Ovarian disorder',
    '720': 'Ankylosing spondylitis',
    '4419': 'Aortic aneurysm',
    '5647': 'Functional bowel disorder',
    '654': 'Complications of pregnancy',
    '5650': 'Anal fissure',
    '476': 'Chronic tonsillitis',
    '5648': 'Irritable bowel syndrome',
    '1853': 'Prostate cancer',
    '40': 'Intestinal obstruction',
    '799': 'Ill-defined condition',
    '2132': 'Benign neoplasm',
    '5651': 'Anal fistula',
    '2266': 'Benign neoplasm thyroid',
    '2170': 'Benign breast disease',
    '188': 'Bladder cancer',
    '3073': 'Adjustment disorder',
    '1071': 'Viral infection',
    '626': 'Menstrual disorder',
    '1521': 'Small intestine cancer',
    '330': 'Degenerative CNS disease',
    '6367': 'Spontaneous abortion',
    '56': 'Urinary condition',
    '4513': 'Phlebitis',
    '1546': 'Rectal cancer',
    '900': 'Injury blood vessels',
    '460': 'Acute nasopharyngitis',
    '1613': 'Laryngeal cancer',
    '484': 'Pneumonia',
    '3456': 'Epilepsy',
    '6384': 'Complicated abortion',
    '6383': 'Complicated pregnancy',
    '3212': 'Meningitis',
    '6154': 'Uterine inflammation',
    '4600': 'Acute respiratory infection'
}

'''
G, diseases, n = read('/Users/sr0215/Python/Clinical/Bayes/Refinement/GENIE_trimmed.gml')
max_weight = max([G[u][v]['weight'] for (u, v) in G.edges()])
for (u, v) in G.edges():
    G[u][v]['weight'] /= max_weight
print (f'There are {len([(u, v) for (u, v) in G.edges() if (v, u) in G.edges()]) / 2} repeats.')

# # GENIE matrix
# A = np.array([[0, 0.1, 0.9, 0],
#               [0.2, 0, 0.1, 0.5],
#               [0, 1.0, 0, 0.1],
#               [0.1, 0.1, 0.9, 0]])

# Calculate row sums and keep dimensions for broadcasting
# row_sums = A.sum(axis=1, keepdims=True)

# Divide to normalize
# A = deepcopy(A / row_sums)
# print (A[:5, :10])

# Generate a sequence for each patient as row and disease tags as columns
Seq = sample(G, None, diseases, l, H, n)

# Seq = []
for seq in Seq:
    for i in range(len(seq)-1):
        u, v = seq[i], seq[i+1]
        if not (G.has_edge(u, v) or u == v):
            print("Mismatch:", u, v)
            break
print(Seq[:10])
# exit(1)

# Apply hierarchical clustering to correlation matrix
# Genes with the same cluster labels are part of the same module
best_label, best_clusters, best_score = None, None, -float('inf')

Xp, Yp = [], []
for nC in range(H - 1, -1, -50):
    model = AgglomerativeClustering(
        n_clusters=nC,
        metric='precomputed',
        linkage='average'
    )

    # Fit the model
    D = dist_matrix(Seq)
    np.fill_diagonal(D, 0)
    clusters = model.fit_predict(D)

    score = silhouette_score(D, clusters, metric='precomputed')
    print (nC, score)

    if score > best_score:
        best_score = score
        best_clusters = clusters
        best_label = nC

    Xp.append(nC)
    Yp.append(score)

plt.plot(Xp, Yp)
plt.axvline(x=best_label, linestyle='dotted')
plt.xlabel('Number of clusters')
plt.ylabel('Calinski-Harabasz score')
plt.title(f'The best cluster-count is {best_label}.')

plt.tight_layout()
plt.savefig('/Users/sr0215/Python/Clinical/Bayes/Refinement/Case_study.png', dpi=150)
# plt.show()

pickle.dump([G, Seq, diseases, n, best_clusters, best_label, Xp, Yp], open('Save_case_study.p', 'wb'))
'''

'''
# Visualize temporal sequence
[G, Seq, diseases, n, best_clusters, best_label, Xp, Yp] = (
    pickle.load(open('Save_case_study.p', 'rb')))

A = pickle.load(open('All_Diseases.p', 'rb'))

most_freq = Counter(best_clusters).most_common(1)[0][0]
for j in range(max(best_clusters)):
    L = [Seq[i] for i in range(len(Seq)) if best_clusters[i] == j]

    D, Timely = freq_temp(L, A)
    All = edge_time_heatmap2(j, D)
    if len(All.keys()) > 20:
        # print(' *** ', j, Timely)
        Pr = {each: sorted(Timely[each], key=Timely[each].get, reverse=True)[0] for each in Timely.keys()}
        print(Pr)
        print(j, {(icd9_dict[Pr[key][0]], icd9_dict[Pr[key][1]])
               for key in Pr.keys()})

        # print({t: (icd9_dict[Timely[each]], icd9_dict[Timely[each][1]]) for t in Pr.keys()})
'''
