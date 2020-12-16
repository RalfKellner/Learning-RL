import gym
import tensorflow as tf
import numpy as np
import random

class cartpole_SARSAAgent():

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
        self.epsilon_decay = 0.9995
        #minimum exploration
        self.epsilon_min = 0.01

        #agent discount
        self.discount = 0.999

        #predicion network for q-values
        self.network = self.build_network()


    def build_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(48, activation = 'relu', input_shape = [self.state_size]),
            tf.keras.layers.Dense(48, activation = 'relu'),
            tf.keras.layers.Dense(self.action_size, activation = 'linear')
        ])

        model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(lr = self.learning_rate))

        return model

    #follow the epsilon-greedy strategy with the current network
    def make_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.network.predict(state)[0])

#make agent
agent = cartpole_SARSAAgent()

#start environment
env = gym.make('CartPole-v1')

#number of episodes
episodes = 500

for e in range(episodes):

    #reset environment
    state = env.reset()
    state = np.reshape(state, (1, agent.state_size))

    #collect the information of the current episodes' return
    sum_episode_steps = 0.0

    while True:

        #collect state, action, reward, state', action', done
        action = agent.make_action(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -5.0
        next_state = np.reshape(next_state, (1, agent.state_size))
        next_action = agent.make_action(next_state)

        sum_episode_steps += 1

        #calculate target value
        target_q = reward
        if not done:
            target_q = reward + agent.discount * agent.network.predict(next_state)[0][next_action]

        #make prediction and replace target value with conducted action q-value
        q_values = agent.network.predict(state)
        q_values[0][action] = target_q

        #learning step with epsilon decay
        agent.network.fit(state, q_values, epochs = 1, verbose = 0)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if done:
            print('Episode {} ended after {} steps, exploration rate: {:.4f}'.format(e, sum_episode_steps, agent.epsilon))
            break

        state = next_state
