import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym

class myPolicy(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_layer = tf.keras.layers.Dense(hidden_dim, activation = 'relu', input_shape = [input_dim], dtype = 'float32')
        self.output_layer = tf.keras.layers.Dense(output_dim, activation = 'softmax', dtype = 'float32')

    def call(self, state):
        x = tf.convert_to_tensor([state], dtype = 'float64')
        x = self.input_layer(x)
        x = self.output_layer(x)
        return x

class Agent():
    def __init__(self, input_dim, hidden_dim, output_dim, gamma, optimizer):
        self.policy = myPolicy(input_dim, hidden_dim, output_dim)
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.gamma = gamma
        self.optimizer = optimizer

    def choose_action(self, state):
        probs = self.policy(state)
        dist = tfp.distributions.Categorical(probs = probs, dtype = tf.float64)
        action = dist.sample()
        self.action_memory.append(action)
        return int(action.numpy()[0])

    def store_reward(self, reward):
        self.reward_memory.append(reward)

    def store_state(self, state):
        self.state_memory.append(state)

    def discounted_rewards(self):
        sum_reward = 0.0
        T = len(self.reward_memory)
        discount_rewards = np.zeros(T)

        for t in reversed(range(T)):
            sum_reward = self.reward_memory[t] + self.gamma * sum_reward
            discount_rewards[t] = (sum_reward)

        return discount_rewards

    def calc_loss(self, prob, action, reward):
        dist = tfp.distributions.Categorical(probs = prob, dtype = 'float32')
        log_prob = dist.log_prob(action)
        loss = -log_prob*reward
        return loss

    def learn(self, discntd_rwrds):
        for state, action, reward in zip(self.state_memory, self.action_memory, discntd_rwrds):
            with tf.GradientTape() as tape:
                p = self.policy(np.array([state]), training=True)
                loss = self.calc_loss(p, action, reward)
                grads = tape.gradient(loss,self.policy.trainable_variables)
                self.optimizer.apply_gradients(zip(grads,self.policy.trainable_variables))

    def reset(self):
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []


env = gym.make('CartPole-v0')


agent = Agent(4, 10, 2, 0.95, tf.keras.optimizers.Adam())

sum_reward = []

for e in range(1000):
    state = env.reset()
    for _ in range(200):
        agent.store_state(state)
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        agent.store_reward(reward)
        if done:
            print('Episode {} ended after {} steps.'.format(e, np.sum(agent.reward_memory)))
            break

    sum_reward.append(np.sum(agent.reward_memory))
    if np.mean(sum_reward[-10:]) > 195:
        print('We reached the aim!')
        break

    discounted_rewards = agent.discounted_rewards()
    agent.learn(discounted_rewards)
    agent.reset()
    del discounted_rewards
