
solvers = ['random', 'dqn2', 'dqnt', 'dqn']
from Solvers.utils import *

def get_solver_class(name):
    if name == solvers[0]:
        from Solvers.Random_Walk import RandomWalk
        return RandomWalk
    elif name == solvers[1]:
        from Solvers.DQN2 import DQN
        return DQN
    elif name == solvers[2]:
        from Solvers.DQNTransfer import DQNT
        return DQNT
    elif name == solvers[3]:
        from Solvers.DQN import DQN
        return DQN
    else:
        assert False, "unknown solver name {}. solver must be from {}".format(name, str(solvers))
