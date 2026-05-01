import random
import numpy as np
import networkx as nx
from copy import deepcopy


def create_gml(VIM3, Diseases):
    G = nx.DiGraph()
    for i in range(len(Diseases)):
        for j in range(len(Diseases)):
            if i == j:
                continue

            G.add_edge(Diseases[i], Diseases[j],
                       weight=VIM3[i, j])
    return G


def spread_reach_chains(G, S, T):

    """
    Simulate an independent cascade diffusion on graph G.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed graph with edge weights as activation probabilities.
    S : list
        Initial seed nodes.
    T : int
        Maximum number of time steps.

    Returns
    -------
    chains : list of lists
        Each chain is a sequence of nodes showing activation path, e.g., [u, v, w].
    """
    chains = []  # store all activation chains
    Seq = deepcopy(S)  # activated nodes
    active_now = deepcopy(S)  # nodes activated in current step
    parent_map = {s: None for s in S}  # track who activated whom

    # record initial seeds as trivial chains
    for s in S:
        chains.append([s])

    t = 0
    while active_now and t < T:
        new_active = []
        for u in active_now:
            for v in G.successors(u):
                if v not in Seq and random.uniform(0, 1) < G[u][v]['weight']:
                    new_active.append(v)
                    parent_map[v] = u  # u activated v

        # form chains for newly activated nodes
        for v in new_active:
            chain = []
            curr = v
            while curr is not None:
                chain.insert(0, curr)
                curr = parent_map[curr]
            chains.append(chain)

        Seq += new_active
        active_now = new_active
        t += 1

    return chains


def permutation_test(l1, l2, num_permutations = 1000):

    # Calculate the observed difference in means
    observed_diff = np.mean(l1) - np.mean(l2)

    # Combine the lists for permutation
    combined = np.concatenate([l1, l2])

    # Initialize an array to store permutation results
    perm_diffs = np.zeros(num_permutations)

    # Perform permutations
    for i in range(num_permutations):
        np.random.shuffle(combined)

        l1_perm = combined[:len(l1)]
        l2_perm = combined[len(l2):]
        perm_diffs[i] = np.mean(l1_perm) - np.mean(l2_perm)

    # Calculate p-value
    p_value = np.mean(perm_diffs >= observed_diff)

    return p_value


def spread_reach_seq(G, S, activation_probability, T):
    # This function simulates an independent cascade–style diffusion
    # process on a directed graph G, starting from an initial seed
    # set S. Over mc=1 Monte Carlo run, newly activated nodes attempt
    # to activate their outgoing neighbors with a fixed activation probability,
    # and the process continues until no new activations occur or the time horizon
    # T is reached.

    Seq, t = deepcopy(S), 0
    new_active = deepcopy(S)

    while len(new_active) > 0 and t < T:
        new_ones = []
        for u in new_active:
            for v in G.successors(u):
                if random.uniform(0, 1) < G[u][v]['weight']:
                    new_ones.append(v)

        new_active = list(set(new_ones) - set(Seq))
        Seq += new_active

        t = t + 1

    return Seq


def spread_reach(G, S, Select, activation_probability, T, mc = 10000):
    # This function simulates an independent cascade–style diffusion
    # process on a directed graph G, starting from an initial seed
    # set S. Over mc Monte Carlo runs, newly activated nodes attempt
    # to activate their outgoing neighbors with a fixed activation probability,
    # and the process continues until no new activations occur or the time horizon
    # T is reached.After each simulation, only activated nodes belonging to a
    # specified subset Select are counted, and both the average number of such
    # activated nodes and their per-node activation frequencies are recorded.
    # The function returns the mean spread over Select and a dictionary estimating
    # each node’s probability of being activated.

    A_all = []
    D = {u: 0 for u in G.nodes()}

    for i in range(mc):
        A, t = deepcopy(S), 0
        new_active = deepcopy(S)

        while len(new_active) > 0 and t < T:
            new_ones = []
            for u in new_active:
                for v in G.successors(u):
                    if random.uniform(0, 1) < activation_probability:
                        new_ones.append(v)

            new_active = list(set(new_ones) - set(A))
            A += new_active
            t = t + 1
            # print ('Number of activated nodes is %d at time %d' % (len(A), t))

        A = [w for w in A if w in Select]
        # print (A)

        A_all.append(len(A))
        for u in A:
            D[u] += 1.0

    D = {u: D[u] / mc for u in D.keys()}

    return np.mean(A_all), D


def remove_edge_outside_ego(H, G, G_E):

    I = G.subgraph([v for v in G.nodes() if v not in list(G_E.nodes())])
    E = list(I.edges())

    if len(E) > 0:
        (u, v) = random.choice(E)
        H.remove_edge(u, v)

    return H

