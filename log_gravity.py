
# Toy script: compare Newtonian vs log-corrected potential (illustrative only)
import numpy as np
def log_gravity_potential(r, G=6.67430e-11, M=1.0, eps=1e-6):
    # V = -GM/r + alpha*log(r)
    alpha = 1e-6
    return -G*M/ (r + eps) + alpha * np.log(r + 1.0)
if __name__ == '__main__':
    import numpy as np
    rs = np.logspace(-2,2,50)
    vals = [log_gravity_potential(r) for r in rs]
    print('sample', vals[:5])
