from unityagents import UnityEnvironment
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

# single agent
# /datasets/home/60/660/aswamina/ece_276c_fa20/RLND-project/p2-continuous-control/Reacher_Linux/Reacher.x86_64
env_path = "Reacher_Linux_NoVis/Reacher.x86_64"
env = UnityEnvironment(file_name=env_path)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

import torch
from collections import deque

# train the agent
def train_ddpg(agent, memory, n_episodes=10, mode='train', 
        actor_pth='./checkpoint/ddpg_actor_checkpoint.pth',
        critic_pth='./checkpoint/ddpg_critic_checkpoint.pth'):
    '''Set up training's configuration and print out episodic performance measures, such as avg scores, avg loss.
    
    Params
    ======
        agent (class object)
        memory (class attribute): agent's attribute for memory size tracking
        mode (string): 'train' or 'test', when in test mode, the agent acts in greedy policy only
        pth (string path): file name for the checkpoint    
    '''
    
    scores = []
    scores_window = deque(maxlen=100)  # last 100 scores
    c_loss_window = deque(maxlen=100)
    a_loss_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment and activate train_mode
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        agent.running_c_loss = 0
        agent.running_a_loss = 0
        agent.training_cnt = 0
        # agent.reset() # reset OUNoise
        
        while True:
            action = agent.act(state, mode)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]      # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished        

            agent.step(state, action, reward, next_state, done)

            score += reward
            state = next_state
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        c_loss_window.append(agent.running_c_loss/(agent.training_cnt+0.0001)) # avoid zero
        a_loss_window.append(agent.running_a_loss/(agent.training_cnt+0.0001)) # avoid zero
        print('\rEpisode {:>4}\tAverage Score:{:>6.3f}\tMemory Size:{:>5}\tCLoss:{:>12.8f}\tALoss:{:>10.6f}'.format(
            i_episode, np.mean(scores_window), len(memory), np.mean(c_loss_window), np.mean(a_loss_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {:>4}\tAverage Score:{:>6.3f}\tMemory Size:{:>5}\tCLoss:{:>12.8f}\tALoss:{:>10.6f}'.format(
                i_episode, np.mean(scores_window), len(memory), np.mean(c_loss_window), np.mean(a_loss_window)))
        if np.mean(scores_window) >= 31:
            break
    torch.save(agent.actor_local.state_dict(), actor_pth)
    torch.save(agent.critic_local.state_dict(), critic_pth)
    return scores

import matplotlib.pyplot as plt
import pandas as pd

def plot_scores(scores, rolling_window=100):
    '''Plot score and its moving average on the same chart.'''
    
    fig = plt.figure(figsize=(10,5))
    plt.plot(np.arange(len(scores)), scores, '-c', label='episode score')
    plt.title('Episodic Score')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(np.arange(len(scores)), rolling_mean, '-y', label='rolling_mean')
    plt.ylabel('score')
    plt.xlabel('episode #')
    plt.legend()

from workspace_utils import active_session
from ddpg_agent import Agent
agent_ddpg = Agent(state_size=33, action_size=4, seed=199)

# with active_session():
ddpg_ep_rewards_list = train_ddpg(agent=agent_ddpg, memory=agent_ddpg.memory, n_episodes=1000)
ddpg_ep_rewards_list = np.array(ddpg_ep_rewards_list)
np.save('./data/ddpg_ep_rewards_list.npy', ddpg_ep_rewards_list)
