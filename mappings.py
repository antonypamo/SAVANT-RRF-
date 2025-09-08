
\"\"\"Mappings for icosahedral and dodecahedral nodal structures with simple semantic embeddings.\"\"\"
import networkx as nx
import math
from sentence_transformers import SentenceTransformer, util

_embedder = SentenceTransformer('all-MiniLM-L6-v2')

class BasePolyMap:
    def __init__(self, G):
        self.G = G
        self.node_texts = {n: f'node_{n}' for n in G.nodes()}
        self.emb = {n: _embedder.encode(self.node_texts[n]) for n in G.nodes()}

    def nodes(self):
        return list(self.G.nodes())

    def closest_node(self, text):
        q = _embedder.encode(text)
        best = None; bsim = -1
        for n, v in self.emb.items():
            sim = util.cos_sim(q, v).item()
            if sim > bsim:
                best, bsim = n, sim
        return best

class IcosaMap(BasePolyMap):
    def __init__(self):
        G = nx.icosahedral_graph()
        super().__init__(G)

class DodecaMap(BasePolyMap):
    def __init__(self):
        G = nx.dodecahedral_graph()
        super().__init__(G)
