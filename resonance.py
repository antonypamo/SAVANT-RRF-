
\"\"\"Resonance simulator: simple coupled-oscillator time series and summary.\"\"\"
import numpy as np
from scipy.signal import periodogram

class ResonanceSimulator:
    def __init__(self):
        pass

    def simulate(self, seed_text, n_nodes=12, steps=512, freq_base=1.0):
        rng = np.random.RandomState(abs(hash(seed_text)) % (2**32))
        A = rng.randn(n_nodes, n_nodes) * 0.05
        A = (A + A.T) * 0.5
        state = rng.randn(n_nodes) * 0.01
        x = np.zeros((steps, n_nodes))
        damping = 0.05
        for t in range(steps):
            input_signal = np.sin(2*np.pi*(freq_base + 0.1*rng.randn()) * (t/steps))
            state = state + 0.1*(A.dot(state) + input_signal) - damping*state
            x[t] = state
        freqs, p = periodogram(x, fs=1.0, axis=0)
        power = p.sum(axis=1)
        dom_idx = int(power.argmax())
        dom_freq = float(freqs[dom_idx])
        return {'summary': {'dom_freq': dom_freq, 'max_power': float(power[dom_idx])},
                'time_series': x.tolist(), 'freqs': freqs.tolist(), 'power': power.tolist()}
