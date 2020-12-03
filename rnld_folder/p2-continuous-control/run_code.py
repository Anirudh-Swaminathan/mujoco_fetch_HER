from unityagents import UnityEnvironment
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import time

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

from ddpg_agent import Agent
from d4pg_agent import AgentD4PG


# define model
DDPG = 0
D4PG = 1
model = DDPG

# define train/test
TRAIN = 0
TEST = 1
train_or_test = TEST

# single agent
env_path = '/home/alexander/RLND-project_Updated/p2-continuous-control/Reacher_Linux/Reacher.x86_64'
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
    if (model == DDPG):
        plt.savefig('DDPG_results.png')
    elif (model == D4PG):
        plt.savefig('DDPG_results.png')


######################## DDPG ################################################
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
   
# test the agent
def test_ddpg(agent, memory, mode='test', 
        actor_pth='./checkpoint/ddpg_actor_checkpoint.pth',
        critic_pth='./checkpoint/ddpg_critic_checkpoint.pth'):
    
    # load models
    agent.actor_local.load_state_dict(torch.load(actor_pth))
    agent.critic_local.load_state_dict(torch.load(critic_pth))
    agent.actor_local.eval()
    agent.critic_local.eval()    

    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    time.sleep(10)                                    # sleep for visualization
    
    while True:
        action = agent.act(state, mode)
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]      # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished        

        agent.step(state, action, reward, next_state, done)
        time.sleep(.1)
        state = next_state
        if done:
            break


# train DDPG model
if (model == DDPG and train_or_test == TRAIN):
    agent_ddpg = Agent(state_size=33, action_size=4, seed=199)
    ddpg_ep_rewards_list = train_ddpg(agent=agent_ddpg, memory=agent_ddpg.memory,
                               n_episodes=5000)
    ddpg_ep_rewards_list = np.array(ddpg_ep_rewards_list)
    np.save('./data/ddpg_ep_rewards_list.npy', ddpg_ep_rewards_list)
    plot_scores(ddpg_ep_rewards_list)
    
# test DDPG model and visualize
if (model == DDPG and train_or_test == TEST):
    # define actor/critic model
    agent_ddpg = Agent(state_size=33, action_size=4, seed=199)
    # test model
    test_ddpg(agent=agent_ddpg, memory=agent_ddpg.memory)

    
    

################################## D4PG #######################################
# train the agent
def train_d4pg(agent, memory, n_episodes=10, mode='train', 
        actor_pth='./checkpoint/d4pg_actor_checkpoint.pth',
        critic_pth='./checkpoint/d4pg_critic_checkpoint.pth'):
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
        state = env_info.vector_observations            # get the current state
        score = 0
        agent.running_c_loss = 0
        agent.running_a_loss = 0
        agent.training_cnt = 0
        # agent.reset() # reset OUNoise
        
        while True:
            action = agent.act(state, mode)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations      # get the next state
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

# test the agent
def test_d4pg(agent, memory, n_episodes, mode='test', 
        actor_pth='./checkpoint/d4pg_actor_checkpoint.pth',
        critic_pth='./checkpoint/d4pg_critic_checkpoint.pth'):
    
    # load models
    agent.actor_local.load_state_dict(torch.load(actor_pth))
    agent.critic_local.load_state_dict(torch.load(critic_pth))
    agent.actor_local.eval()
    agent.critic_local.eval()    

    for i in range(10):
        env_info = env.reset(train_mode=False)[brain_name]
        time.sleep(.1)

    for i in range(n_episodes):
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        state = env_info.vector_observations[0]           # get the current state
        time.sleep(5)                                    # sleep for visualization
        
        while True:
            action = agent.act(state, mode)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished        
    
            agent.step(state, action, reward, next_state, done)
            #time.sleep(.1)
            state = next_state
            if done:
                break

# train d4pg
if (model == D4PG and train_or_test == TRAIN):
    agent_d4pg = AgentD4PG(state_size=33, action_size=4, seed=168)
    #from IPython.core.debugger import set_trace
    #set_trace()
    d4pg_ep_rewards_list = train_d4pg(agent=agent_d4pg, memory=agent_d4pg.memory,
                                n_episodes=10000,
                                mode = "train",
                                actor_pth='./checkpoint/d4pg_actor_checkpoint.pth',
                                critic_pth='./checkpoint/d4pg_critic_checkpoint.pth')
    d4pg_ep_rewards_list = np.array(d4pg_ep_rewards_list)
    np.save('./data/d4pg_ep_rewards_list.npy', d4pg_ep_rewards_list)
    plot_scores(d4pg_ep_rewards_list)

# test D4PG model and visualize
if (model == D4PG and train_or_test == TEST): 
    # define actor/critic model
    agent_d4pg = AgentD4PG(state_size=33, action_size=4, seed=68)
    # test model
    test_d4pg(agent=agent_d4pg, memory=agent_d4pg.memory, n_episodes=1)