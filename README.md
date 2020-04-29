CS689 Final Project: Using Transfer Learning to train a Reinforcement learning agent to learn different similar looking environments

This project is a part of CSCE 680 Final project submission. The main Goal of this project is to train an agent to play different levels of Sonic. This main motivation behnd this project is OpenAI's retro contest. WE have use transfer learning to implement a DQN and have tested it across multiple levels. The agent is able to unerstand multiple levels.

Setup:
pip install gym gym-retro
git clone --recursive https://github.com/openai/retro-contest.git
pip install -e "retro-contest/support[docker,rest]"
apt-get install python-opengl -y
apt install xvfb -y
pip install pyvirtualdisplay piglet

Purchase Sonic The HedgeHog from steam and import the ROM from the steam folder.
python -m retro.import <pathtosteamfolder>

Use run.py to execute the code.

These are the arguments used in the project.
Solvers available are
solvers = ['random', 'dqn2', 'dqnt', 'dqn']

To test on a game/domain and a level use this command 
python run.py -s "dqnt" -d <GameName> -l <LevelName> -i [128,128] -a 0.001 -g 0.95 -p 0.05 -P 0.01 -c 0.95 -m 100 -N 5000 -b 32 -G 1 -t 3000 -e 0 -G 1

use level_data to find all the supported levels.

python run.py -s "dqnt" -d "SonicTheHedgehog-Genesis" -l "GreenHillZone.Act2" -i [128,128] -a 0.001 -g 0.95 -p 0.05 -P 0.01 -c 0.95 -m 100 -N 5000 -b 32 -G 1 -t 3000 -e 0 -G 1

python run.py -s "dqnt" -d "SonicAndKnuckles3-Genesis" -l "CarnivalNightZone.Act2" -i [128,128] -a 0.001 -g 0.95 -p 0.05 -P 0.01 -c 0.95 -m 100 -N 5000 -b 32 -G 1 -t 3000 -e 0 -G 1

to train for 100 epochs:
python run.py -s "dqnt" -d "SonicTheHedgehog-Genesis" -l "GreenHillZone.Act2" -i [128,128] -a 0.001 -g 0.95 -p 0.05 -P 0.01 -c 0.95 -m 100 -N 5000 -b 32 -G 1 -t 3000 -e 100


-e is used to give the number of epochs to train. If 0, this will execute the already trained model.
-G 1 is used to display/render the output.

Run a specified RL algorithm on a specified domain.

Options:
  -h, --help            show this help message and exit
  -s SOLVER, --solver=SOLVER
                        Solver from ['random', 'dqn2', 'dqnt', 'dqn']
  -d DOMAIN, --domain=DOMAIN
                        Domain from OpenAI Retro Gym
  -l LEVEL, --level=LEVEL
                        Level from domain in OpenAI Retro Gym
  -x FILE, --experiment_dir=FILE
                        Directory to save Tensorflow summaries in
  -e EPISODES, --episodes=EPISODES
                        Number of episodes for training
  -t STEPS, --steps=STEPS
                        Maximal number of steps per episode
  -i INPUT_SIZE, --input_size=INPUT_SIZE
                        size that the input will be resized to for processing
                        data by the neural network
  -a ALPHA, --alpha=ALPHA
                        The learning rate (alpha) for updating state/action
                        values
  -r SEED, --seed=SEED  Seed integer for random stream
  -G GRAPHICS_EVERY, --graphics=GRAPHICS_EVERY
                        Graphic rendering every i episodes. i=0 will present
                        only one, post training episode.i=-1 will turn off
                        graphics. i=1 will present all episodes.
  -g GAMMA, --gamma=GAMMA
                        The discount factor (gamma)
  -p EPSILON, --epsilon=EPSILON
                        Initial epsilon for epsilon greedy policies (might
                        decay over time)
  -P EPSILON_END, --final_epsilon=EPSILON_END
                        The final minimum value of epsilon after decaying is
                        done
  -c EPSILON_DECAY, --decay=EPSILON_DECAY
                        Epsilon decay factor
  -m REPLAY_MEMORY_SIZE, --replay=REPLAY_MEMORY_SIZE
                        Size of the replay memory
  -N UPDATE_TARGET_ESTIMATOR_EVERY, --update=UPDATE_TARGET_ESTIMATOR_EVERY
                        Copy parameters from the Q estimator to the target
                        estimator every N steps.
  -b BATCH_SIZE, --batch_size=BATCH_SIZE
                        Size of batches to sample from the replay memory
  -z SKIPZEROCOUNT, --skipzerocount=SKIPZEROCOUNT
                        continous zero rewards after which to start skipping