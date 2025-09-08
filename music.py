
\"\"\"Music adapter: map text to simple MIDI-like sequences (pitch,duration).\"\"\"
import numpy as np

class MusicAdapter:
    def __init__(self):
        self.scale = [0,2,4,5,7,9,11]  # major
        self.base = 60

    def adapt_text_to_music(self, text, length=16):
        h = abs(hash(text))
        rng = np.random.RandomState(h % (2**32))
        seq = []
        for i in range(length):
            step = rng.randint(0, len(self.scale))
            pitch = self.base + self.scale[step] + rng.randint(-2,2)
            dur = float(rng.choice([0.25,0.5,1.0]))
            seq.append((int(pitch), dur))
        return seq
