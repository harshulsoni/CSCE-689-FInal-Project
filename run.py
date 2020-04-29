import gym
import optparse
import sys
import os
import random
import numpy as np
#import retro
from retro_contest.local import make
#import retro

if "../" not in sys.path:
  sys.path.append("../")

from Solvers.Abstract_Solver import AbstractSolver, Statistics
import Solvers.Available_solvers as avs
from Solvers.utils import *


def readCommand(argv):
    parser = optparse.OptionParser(description='Run a specified RL algorithm on a specified domain.')
    parser.add_option("-s", "--solver", dest="solver", type="string", default="sarsa",
                      help='Solver from ' + str(avs.solvers))
    parser.add_option("-d", "--domain", dest="domain", type="string", default="SonicAndKnuckles3-Genesis",
                      help='Domain from OpenAI Retro Gym')
    parser.add_option("-l", "--level", dest="level", type="string", default="CarnivalNightZone.Act2",
                      help='Level from domain in OpenAI Retro Gym')
    parser.add_option("-x", "--experiment_dir", dest="experiment_dir", default="Experiment",
                      help="Directory to save Tensorflow summaries in", metavar="FILE")
    parser.add_option("-e", "--episodes", type="int", dest="episodes", default=500,
                      help='Number of episodes for training')
    parser.add_option("-t", "--steps", type="int", dest="steps", default=1000,
                      help='Maximal number of steps per episode')
    parser.add_option("-i", "--input_size", dest="input_size", type="string", default="[64,64]",
                      help='size that the input will be resized to for processing data by the neural network')
    parser.add_option("-a", "--alpha", dest="alpha", type="float", default=0.5,
                      help='The learning rate (alpha) for updating state/action values')
    parser.add_option("-r", "--seed", type="int", dest="seed", default=random.randint(0, 9999999999),
                      help='Seed integer for random stream')
    parser.add_option("-G", "--graphics", type="int", dest="graphics_every", default=0,
                      help='Graphic rendering every i episodes. i=0 will present only one, post training episode.'
                           'i=-1 will turn off graphics. i=1 will present all episodes.')
    parser.add_option("-g", "--gamma", dest="gamma", type="float", default=1.00,
                      help='The discount factor (gamma)')
    parser.add_option("-p", "--epsilon", dest="epsilon", type="float", default=0.1,
                      help='Initial epsilon for epsilon greedy policies (might decay over time)')
    parser.add_option("-P", "--final_epsilon", dest="epsilon_end", type="float", default=0.1,
                      help='The final minimum value of epsilon after decaying is done')
    parser.add_option("-c", "--decay", dest="epsilon_decay", type="float", default=0.99,
                                        help='Epsilon decay factor')
    parser.add_option("-m", "--replay", type="int", dest="replay_memory_size", default=500000,
                      help='Size of the replay memory')
    parser.add_option("-N", "--update", type="int", dest="update_target_estimator_every", default=10000,
                      help='Copy parameters from the Q estimator to the target estimator every N steps.')
    parser.add_option("-b", "--batch_size", type="int", dest="batch_size", default=32,
                      help='Size of batches to sample from the replay memory')
    parser.add_option("-z", "--skipzerocount", type="int", dest="skipzerocount", default=100,
                      help='continous zero rewards after which to start skipping')
    (options, args) = parser.parse_args(argv)
    return options

def getEnv(domain, level, solver):
    # try:
        #return retro.make(game=domain, state=level, scenario="contest")

    solvers = ['random', 'dqn2', 'dqnt', 'dqn']
    
    env = make(game=domain, state=level)

    if solver == solvers[0]:
        env =  AllowBacktracking(Discretizer(env)) 
    elif solver == solvers[1]:
        env = AllowBacktracking2(Discretizer2(env))
    elif solver == solvers[2]:
        env = AllowBacktracking(Discretizer(env))
    elif solver == solvers[3]:
        env = AllowBacktracking(Discretizer2(env))
    else:
        assert False, "unknown solver name {}. solver must be from {}".format(name, str(solvers))
    
    return env
    # except:
    #     assert False, "Domain and level must be a valid (and installed) Gym Retro environment"


def parse_list(string):
    string = string[1:-1].split(',') # Change "[0,1,2,3]" to '0', '1', '2', '3'
    l = []
    for n in string:
        l.append(int(n))
    return tuple(l)



if __name__ == "__main__":
    options = readCommand(sys.argv)
    resultdir = "Results/"
    resultdir = os.path.abspath("./{}".format(resultdir))
    options.experiment_dir = os.path.abspath("./{}".format(options.experiment_dir))

    # Create result file if one doesn't exist
    # print(os.path.join(resultdir, options.outfile + '.csv'))
    # if not os.path.exists(os.path.join(resultdir, options.outfile + '.csv')):
    #     with open(os.path.join(resultdir, options.outfile + '.csv'), 'w+') as result_file:
    #         result_file.write(AbstractSolver.get_out_header())

    random.seed(options.seed)
    env = getEnv(options.domain, options.level, options.solver)
    # env = Discretizer(env)
    # env = AllowBacktracking(env)
    #env = RewardScaler(env)
    env._max_episode_steps = options.steps
    print("Domain action space is {}".format(env.action_space))
    try:
        options.input_size = parse_list(options.input_size)
    except ValueError:
        raise Exception('input size argument doesnt follow int array conventions i.e., [<int>,<int>]')
    except:
        pass
    solver = avs.get_solver_class(options.solver)(env,options)

    for i_episode in range(options.episodes):
        solver.render = False
        if options.graphics_every > 0 and i_episode % options.graphics_every == 0:
            solver.render = True
        solver.init_stats()
        solver.statistics[Statistics.Episode.value] += 1
        env.reset()
        solver.train_episode()
        # result_file.write(solver.get_stat() + '\n')
        # Decay epsilon
        if options.epsilon > options.epsilon_end:
            options.epsilon *= options.epsilon_decay
        # Update statistics
        # stats.episode_rewards[i_episode] = solver.statistics[Statistics.Rewards.value]
        # stats.episode_lengths[i_episode] = solver.statistics[Statistics.Steps.value]
        print("Episode {}: Reward {}, Steps {}".format(i_episode+1,solver.statistics[Statistics.Rewards.value],
                                                       solver.statistics[Statistics.Steps.value]))
    if options.graphics_every > -1:
        solver.render = True
        solver.run_greedy()
    # solver.plot(stats)
    solver.close()
