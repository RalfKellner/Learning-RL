import autograd.numpy as np
from autograd import grad, elementwise_grad
import random
import gym

def approx(weights, state, action):
    return np.matmul(state, weights)[action]

def policy(weights, state, epsilon):
    actions = [0, 1]
    if np.random.rand() < epsilon:
        return random.choice(actions)
    else:
        possible_vals = []
        for a in actions:
            possible_vals.append(approx(weights, state, a))

        return np.argmax(possible_vals)

epsilon = 0.20
weights = np.random.normal(size = 4 * 2).reshape(4, 2)
dapprox = grad(approx)
episodes = 200

discount = 0.99
alpha = 0.10

#intialize gym environment
env = gym.make('CartPole-v0')
rewards = []
for e in range(episodes):

    state = env.reset()
    sum_rewards = 0.0
    for _ in range(200):

        action = policy(weights, state, epsilon)
        state_, reward, done, _ = env.step(action)
        q_hat = approx(weights, state, action)
        q_hat_grad = dapprox(weights, state, action)
        sum_rewards += 1
        if sum_rewards >= 200:
            print('Episode {} ended after 200 steps.'.format(e))

        if done:
            print('Episode {} ended after {} steps.'.format(e, sum_rewards))
            weights += alpha*(reward - q_hat) * q_hat_grad
            break
        else:
            action_ = policy(weights, state_, epsilon)
            q_hat_ = approx(weights, state_, action_)
            weights += alpha*(reward - discount*q_hat_) * q_hat_grad
            state = state_

    rewards.append(sum_rewards)
