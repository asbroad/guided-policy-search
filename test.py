#load_ext autoreload
#autoreload 2
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

floatX = theano.config.floatX
from functools import partial
from collections import OrderedDict

from numpy import pi

from IPython.display import clear_output

from guided.problem import DoublePendulumProblem
from guided.tmodel import PolicyModel, MultiStepPolicyModel, TemporalMultiStepPolicyModel
import ipython_animations
ipython_animations.enable_inline()

def animation_for_model(model, initial_state, time_horizon, path=None):
    problem = DoublePendulumProblem()
    plant = problem.plant()
    plant.controller = model.controller(model)
    #plant.controller = lambda x,t : 0.0
    traj = problem.compute_trajectory(plant, initial_state, time_horizon)
    problem.plot_u(plant, traj)
    print('Sampled trajectory of cost %f.' % (problem.cost(traj),))
    animation = problem.animate_trajectory(plant, traj)
    if path is not None:
        animation.save(path, fps=20)
        animation._save_path = path
    return animation

problem = DoublePendulumProblem()
plant = problem.plant()

#model = PolicyModel.from_plant(plant, internal_layers=[100, 200], dropout=0.2)

# tries to sort of keep it up
#model = MultiStepPolicyModel.from_plant(plant, internal_layers=[50], dropout=0.2,)

# with timestep 3, and no dropout, about same
# model = MultiStepPolicyModel.from_plant(plant, internal_layers=[50], dropout=0.,)

# with timestep 7, and no dropout, about same
#model = MultiStepPolicyModel.from_plant(plant, internal_layers=[50], dropout=0.,)

# with timestep 3, no dropout, and error for all timesteps
#model = MultiStepPolicyModel.from_plant(plant, internal_layers=[50], dropout=0.,)

# with timestep 10, no dropout, and error for all timesteps, and policy
#model = MultiStepPolicyModel.from_plant(plant,
#                                                 internal_layers=[50],
#                                                 dropout=0.,
#                                                 penalize_over_trajectory = True,
#                                                 policy_laziness=0.001,
#                                                 num_steps = 10)

# with timestep 10, no dropout, and error for last timestep, and policy cost
#model = MultiStepPolicyModel.from_plant(plant,
#                                                 internal_layers=[50],
#                                                 dropout=0.,
#                                                 penalize_over_trajectory = True,
#                                                 policy_laziness=0.0001,
#                                                 num_steps = 3)
#
# with linear activation at input, mod_by_pi, and 5 steps, penalize at end.
# with softrelu inside
#model = MultiStepPolicyModel.from_plant(plant,
#                                                 internal_layers=[50, 50],
#                                                 dropout=0.1,
#                                                 mod_by_pi = True,
#                                                 penalize_over_trajectory = False,
#                                                 policy_laziness=0.01,
#                                                 num_steps = 5)
#
model = TemporalMultiStepPolicyModel.from_plant(plant,
                                                 hidden_size = 20,
                                                 internal_layers=[50, 50],
                                                 dropout=0.1,
                                                 mod_by_pi = True,
                                                 penalize_over_trajectory = False,
                                                 policy_laziness=0.01,
                                                 num_steps = 5)
