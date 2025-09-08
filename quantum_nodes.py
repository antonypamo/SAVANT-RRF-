
# Toy simulation: random superposition over nodal amplitudes
import numpy as np
def random_superposition(n=12):
    psi = np.random.randn(n) + 1j * np.random.randn(n)
    psi /= np.linalg.norm(psi)
    probs = np.abs(psi)**2
    return probs
if __name__ == '__main__':
    print(random_superposition(12))
