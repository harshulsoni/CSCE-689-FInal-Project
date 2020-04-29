# python "run.py" -s "dqn" -d "SonicAndKnuckles3-Genesis" -l "CarnivalNightZone.Act1" -i [64,64] -a 0.001 -g 0.95 -p 0.25 -P 0.02 -c 0.95 -m 1000 -N 100 -b 64 -t 5000 -e 0 -z 500

import os
import random
from collections import deque
import tensorflow as tf
from keras import backend as bk
from keras.layers import Dense, Lambda
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import huber_loss
import keras
# from keras.callbacks import TensorBoard
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape, MaxPooling2D
from keras.models import load_model
from keras.models import model_from_json
from skimage.transform import resize
import time
from datetime import datetime

from Solvers.Abstract_Solver import AbstractSolver


class DQN(AbstractSolver):
    def __init__(self,env,options):
        # assert (str(env.action_space).startswith('Discrete') or
        # str(env.action_space).startswith('Tuple(Discrete')), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)
        self.model = self._build_model()
        self.target_model = self._build_model()
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        # Add required fields          #
        ################################
        # initializing the memory array
        self.D = []
        self.steps = 0
        state = self.env.reset()
        self.policy = self.make_epsilon_greedy_policy()
        # self.policy = self.create_greedy_policy()
        
        self.update_model = False
        nA = self.env.action_space.n
        epsilon = self.options.epsilon
        for _ in range(self.options.replay_memory_size):
            action_probs = self.policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = self.step(action)
            self.env.render()
            time.sleep(0.15)
            self.D.append((state, action , reward, next_state, done))
            state = next_state
            if done:
                state = self.env.reset()

 
    def _build_model(self):
        input_size = self.options.input_size
        action_size = self.env.action_space.n
        # layers = self.options.layers
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        
        # ______Model 4
        model.add(Lambda(lambda image: tf.image.resize(image, input_size)))
        model.add(Conv2D(1, kernel_size=(3, 3)))
        model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(action_size))
        


        # tensorboard = TensorBoard(log_dir="logs/sonic")
        # tensorboard.set_model(model)

        if os.path.exists('sonic_model.json'):
            print("Model Found")
            with open('sonic_model.json', 'r') as model_file:
                loaded_model_json = model_file.read()
                model = model_from_json(loaded_model_json, custom_objects={'tf': tf})
                # load weights into new model
                model.load_weights("sonic_model.h5")
            print("Model Loaded")
        model.compile(loss=huber_loss,
                      optimizer=Adam(lr=self.options.alpha))

        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
        # serialize model to JSON
        model_json = self.model.to_json()
        now = datetime.now()
        with open("sonic_model2.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        self.model.save_weights("sonic_model2.h5")
        # self.model.save_weights(f"sonic_model2_{date_time}.h5")
        print('Model Saved')
        print('Model Saved')
        print('Model Saved')
        print('Model Saved')
        print('Model Saved')

    def make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on the given Q-function approximator and epsilon.

        Returns:
            A function that takes a state as input and returns a vector
            of action probabilities.
        """
        nA = self.env.action_space.n
        epsilon = self.options.epsilon
        input_size = self.options.input_size
        def policy_fn(state):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            state_arr = np.array([state])
            q_values =  self.model.predict(state_arr)[0]
            actions_prob = np.ones(nA)/nA * epsilon
            best_action = np.argmax(q_values)
            actions_prob[best_action] = 1 - epsilon + (epsilon / nA)
            return actions_prob
            #raise NotImplementedError

        return policy_fn

    def train_episode(self):
        """
        Perform a single episode of the Q-Learning algorithm for off-policy TD
        control using a DNN Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Use:
            self.options.experiment_dir: Directory to save DNN summaries in (optional)
            self.options.replay_memory_size: Size of the replay memory
            self.options.update_target_estimator_every: Copy parameters from the Q estimator to the
              target estimator every N steps
            self.options.batch_size: Size of batches to sample from the replay memory
            self.env: OpenAI environment.
            self.options.gamma: Gamma discount factor.
            self.options.epsilon: Chance the sample a random action. Float betwen 0 and 1.
            new_state, reward, done, _ = self.step(action): To advance one step in the environment
            state_size = self.env.observation_space.shape[0]
            self.model: Q network
            self.target_model: target Q network
            self.update_target_model(): update target network weights = Q network weights
        """

        # Reset the environment
        state = self.env.reset()
        nA = self.env.action_space.n
        alpha = self.options.alpha
        gamma = self.options.gamma
        policy = self.make_epsilon_greedy_policy()
        replay_memory_size = self.options.replay_memory_size
        batch_size = self.options.batch_size
        epsilon = self.options.epsilon
        input_size = self.options.input_size

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        zerocount = 0
        for step in range(self.options.steps):
        #while True:
            self.steps += 1
            if self.steps % self.options.update_target_estimator_every == 0:
                self.update_model = True
            action_probs = policy(state)

            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            # action = 1
            next_state, reward, done, _ = self.step(action)
            # ______________________________________________________________
            self.env.render()
            if reward == 0:
                zerocount += 1
            else:
                zerocount = 0
            if zerocount > self.options.skipzerocount:
                reward = self.options.skipzerocount - zerocount
                
                continue
            self.D = self.D[1:]
            self.D.append((state, action , reward, next_state, done))
            batch = random.sample(self.D, batch_size)
            random.shuffle(batch)
            batch = np.array(batch)
            next_state_arr = np.array([list(x) for x in batch[:, 3]])
            state_arr = np.array([list(x) for x in batch[:, 0]])
            action_arr = np.array([int(x) for x in batch[:, 1]])
            reward_arr = batch[:, 2]
            done_arr = batch[:, 4]
            y_arr = self.model.predict(state_arr)

            not_done_arr = 1 - done_arr
            next_action_values = self.target_model.predict(next_state_arr)
            next_action_arr = np.argmax(next_action_values, axis=1)
            
            # single line code.
            # reward_arr += done_arr * gamma * np.multiply(next_action_values, np.eye(batch_size)[next_action_arr])
            
            # multi line of the above single line. To test properly
            next_state_best_action_value = next_action_values[np.arange(len(next_action_arr)), next_action_arr]
            additional_reward = np.multiply(not_done_arr, gamma * next_state_best_action_value)
            reward_arr += additional_reward
            y_arr[np.arange(len(action_arr)), action_arr] = reward_arr
            self.model.fit(state_arr, y_arr, verbose=0)

            # same as above but iteratively. so slow. (very very slow)
            # for (statej, actionj , rewardj, next_statej, donej, yj) in batch:
            #     if not donej:
            #         next_statej = np.array([next_statej])
            #         next_state_values = self.target_model.predict(next_statej)[0]
            #         best_next_action = np.argmax(next_state_values)
            #         best_next_action_value = next_state_values[best_next_action]
            #         rewardj += gamma * best_next_action_value
            #     # statej_arr = np.array([statej])
            #     # yj = self.model.predict(statej_arr)[0]
            #     yj[actionj] = rewardj
            #     x.append(statej)
            #     y.append(yj)
            # y = np.array(y)
            # x = np.array(x)
            # self.model.fit(x, y, verbose=0)

            if self.update_model:
                print("Updating Model")
                self.update_target_model()
                self.update_model = False
            print(f"Step {step} completed with reward {reward} with action {action}")
            state = next_state
            if done:
                break
        #print("Total Steps:", self.steps)
        #raise NotImplementedError

    def __str__(self):
        return "DQN"

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.


        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """
        nA = self.env.action_space.n
        def policy_fn(state):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            state_arr = np.array([state])
            q_values =  self.model.predict(state_arr)[0]
            actions_prob = np.zeros(nA)
            #print(len(actions_prob))
            best_action = np.argmax(q_values)
            actions_prob[best_action] = 1
            return actions_prob
            #raise NotImplementedError

        return policy_fn
