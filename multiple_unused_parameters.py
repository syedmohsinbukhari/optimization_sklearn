import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args


np.random.seed(0)


space = [Real(-4, 4, name='x'), ]
static_params = {'y': 3}


def f(**params):
    _x = np.array([params['x']]) + (np.random.rand()*0.03) - 0.015
    _y = np.array([params['y']]) + (np.random.rand()*0.03) - 0.015
    _f = ((_x[0]-2) ** 2) + ((_y[0]-3) ** 2)
    print(f"{_f} = (({_x[0]}-2)^2) + (({_y[0]}-3)^2)")
    return _f


@use_named_args(space)
def objective(**params):
    all_params = {**params, **static_params}
    return f(**all_params)


print("Minimizing objective")
res = gp_minimize(objective, space, n_calls=50, random_state=0)
print("Done minimizing")
print("Best parameters")
print(f"x={res.x[0]}")
