
\"\"\"Core orchestration engine for SAVANT-RRF (simplified demo implementation).
\"\"\"
import random
from .mappings import IcosaMap, DodecaMap
from .resonance import ResonanceSimulator
from .music import MusicAdapter
from .memory import MemoryStore

class SavantEngine:
    def __init__(self):
        self.memory = MemoryStore("datasets/memory.jsonl")
        self.icosa = IcosaMap()
        self.dodeca = DodecaMap()
        self.res_sim = ResonanceSimulator()
        self.music = MusicAdapter()

    def _classify(self, text):
        t = text.lower()
        if any(k in t for k in ('reson', 'frecuen', 'sinton')):
            return 'resonance'
        if any(k in t for k in ('mus', 'nota', 'melod')):
            return 'music'
        return 'map'

    def handle_query(self, text):
        func = self._classify(text)
        if func == 'resonance':
            r = self.res_sim.simulate(text)
            self.memory.add({'type':'resonance','query':text,'result':r['summary']})
            return f\"[resonance] dom_freq={r['summary']['dom_freq']:.4f} max_power={r['summary']['max_power']:.4f}\"
        elif func == 'music':
            seq = self.music.adapt_text_to_music(text)
            self.memory.add({'type':'music','query':text,'result':{'len':len(seq)}})
            return f\"[music] generated sequence length={len(seq)} (sample={seq[:3]})\"
        else:
            node = self.icosa.closest_node(text)
            self.memory.add({'type':'map','query':text,'node':str(node)})
            return f\"[mapped -> icosa_node:{node}]\"
