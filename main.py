import sys
import gym
import numpy as np
from collections import defaultdict

from plot_utils import plot_blackjack_values, plot_policy

def episode_eps_greedy(env, eps, policy):
    episode = []
    state = env.reset()
    while True:
        if state in policy:
            if np.random.random() > eps:
                action = policy[state]
            else:
                probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
                action = np.random.choice(np.arange(2), p=probs)
        else:
            probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
            action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

def mc_control(env, num_episodes, alpha, gamma=1.0, eps=0.8, min_eps=0.05):
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    policy = {}
    for i_episode in range(1, num_episodes+1):
        if i_episode % 1000 == 0:
            eps *= 0.98
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
        episode = episode_eps_greedy(env, max(eps, min_eps), policy)
        states, actions, rewards = zip(*episode)
        G = 0
        for i in range(len(episode)):
            G += rewards[i] * (gamma**i)
            Q[states[i]][actions[i]] = ((1 - alpha)*Q[states[i]][actions[i]]) + (alpha * G)
        for s in Q:
            policy.update({s : np.argmax(Q[s])})
    return policy, Q

env = gym.make('Blackjack-v0')
print(env.observation_space)
print(env.action_space)

policy, Q = mc_control(env, 200000, 0.05)

V = dict((k,np.max(v)) for k, v in Q.items())

plot_blackjack_values(V)
plot_policy(policy)
