import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args


np.random.seed(0)


space = [Real(-4, 4, name='x'),
         Real(-4, 4, name='y')]


@use_named_args(space)
def f(**params):
    x = np.array([params['x']]) + (np.random.rand()*0.03) - 0.015
    y = np.array([params['y']]) + (np.random.rand()*0.03) - 0.015
    return ((x[0]-2) ** 2) + ((y[0]-3) ** 2)


print("Minimizing objective")
res = gp_minimize(f, space, n_calls=50, random_state=0)
print("Done minimizing")
print("Best parameters")
print(f"x={res.x[0]}")
print(f"y={res.x[1]}")
