
# Toy resonance universe simulation orchestrating many seeds
from core.resonance import ResonanceSimulator
def run_many(N=5):
    sim = ResonanceSimulator()
    for i in range(N):
        r = sim.simulate(f'seed_{i}')
        print('seed', i, 'dom_freq', r['summary']['dom_freq'])
if __name__ == '__main__':
    run_many(3)
