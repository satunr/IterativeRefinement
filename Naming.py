import networkx as nx
import pickle


'''
mapping = {}
repeat = 0
with open("GENIE.gml", "r") as f:

    current_id = None

    for line in f:
        line = line.strip()

        if line.startswith("id"):
            current_id = int(line.split()[1])

        elif line.startswith("label"):
            label = line.split()[1].strip('"')
            mapping[current_id] = label

        # if repeat >= 100:
        #         break

        repeat = repeat + 1

print(mapping)
pickle.dump(mapping, open('mapping.p', 'wb'))
'''

mapping = pickle.load(open('mapping.p', 'rb'))
mapping = {str(u): mapping[u] for u in mapping.keys()}
print (mapping)

G0 = pickle.load(open('GENIE.p', 'rb'))
print(G0.nodes())

G = nx.read_gml('GENIE_trimmed.gml')
print ([(u, v) for (u, v) in G.edges()
        if (mapping[u], mapping[v]) not in G0.edges() and u in G0.nodes() and v in G0.nodes()])
# print (len(G.edges()))