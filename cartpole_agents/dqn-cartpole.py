import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque



class cartpoleDQAgent():

    def __init__(self):

        #input dimension
        self.state_size = 4
        #output dimension
        self.action_size = 2

        #learning rate for the optimizer
        self.learning_rate = 0.005

        #fraction for epsilon greedy strategy
        self.epsilon = 1.0
        #decay rate for epsilon
        self.epsilon_decay = 0.999
        #minimum exploration
        self.epsilon_min = 0.01

        #stacked memory for s, a, r, s', done experiences
        self.memory = deque(maxlen = 5000)

        #learning examples after every move
        self.batch_size = 32

        #agent discount
        self.discount = 0.99

        #after this number of actions, the target networks receives weights
        #from the prediction network
        self.update_targetsteps = 20

        #prediction and target network: input - state - output - two q-values for actions 0 and 1
        self.prediction_model = self.build_network()
        self.target_model = self.build_network()

    #function for building a neural network
    def build_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(48, activation = 'selu', input_shape = [self.state_size]),
            tf.keras.layers.Dense(64, activation = 'selu'),
            tf.keras.layers.Dense(self.action_size, activation = 'linear')
        ])
        model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(lr = self.learning_rate))
        return model

    #receive learned weights from the prediction network
    def set_target_weights(self):
        self.target_model.set_weights(self.prediction_model.get_weights())

    #follow the epsilon-greedy strategy with the current network
    def make_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.prediction_model.predict(state)[0])

    #store s, a, r, s', done experience
    def build_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    #heart of the function, draw a batch from the memory
    #for every element in the batch, create the target value and
    #use it to replace the respective value of the prediction network
    #basically at time = now determine which would have been best with knowledge at t = next_time_step
    #insert this information in the current q-value because this is what the prediction network
    #is supposed to learn
    def replay_learn(self):
        if len(self.memory) < self.batch_size:
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target_q = reward
            if not done:
                target_q = reward + self.discount * np.amax(self.target_model.predict(next_state)[0])

            q_values = self.prediction_model.predict(state)
            q_values[0][action] = target_q
            self.prediction_model.fit(state, q_values, epochs = 1, verbose = 0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



#start the environment
env = gym.make('CartPole-v1')

#initialize the agent
agent = cartpoleDQAgent()

#collect the number of all time steps played
#this is needed as we will update the target network's weights
#after every update_targetsteps
overall_timesteps = 0.0

#number of episodes we want to play/learn
episodes = 50

for e in range(episodes):
    #reset the cartpole
    state = env.reset()
    state = np.reshape(state, (1, agent.state_size))

    #collect the information of the current episodes' return
    sum_episode_steps = 0.0
    while True:

        #collect s, a, r, s' experiences
        action = agent.make_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, (1, agent.state_size))
        reward = reward if not done else -5.0
        agent.build_memory(state, action, reward, next_state, done)
        sum_episode_steps += 1
        overall_timesteps += 1

        #maybe update the target network
        if overall_timesteps % agent.update_targetsteps == 0:
            agent.set_target_weights()

        #if not done, use the next state as the current state and make a move
        state = next_state
        if done:
            print('Episode {} ended after {} steps, exploration rate: {:.4f}'.format(e, sum_episode_steps, agent.epsilon))
            break

        #at every step played, the prediction network learns with a batch of size: batch_size
        agent.replay_learn()
