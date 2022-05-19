import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np

from networkreplay import DeepQNetwork, ReplayBuffer

class Agent():
    def __init__(self, type, gamma, lr, n_actions, n_states, 
                batch_size, mem_size, replace, saved_dir, env_name):

        self.type = type        
        self.gamma = gamma
        self.action_space = np.arange(n_actions)
        self.batch_size = batch_size
        self.replace_num = replace
        self.epsilon = 1.0
        self.learn_idx = 0
        self.loss_plot = 0
        self.running_loss = 0
        
        self.memory = ReplayBuffer(mem_size, n_states)
        
        self.Q_eval = DeepQNetwork(lr=lr,out_dims=n_actions,input_dims=n_states,
                                   name=env_name+'.pth', saved_dir=saved_dir, 
                                   fc1_dims=128, fc2_dims=64)
        self.Q_next = DeepQNetwork(lr=lr,out_dims=n_actions,input_dims=n_states,
                                   name=env_name+'_q_next.pth', saved_dir=saved_dir, 
                                   fc1_dims=128, fc2_dims=64)
        
        
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array(observation)).to(self.Q_eval.device)
            action = T.argmax(self.Q_eval.forward(state)).item()
        else:
            action = np.random.choice(self.action_space)
            
        return action    
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def save_models(self):
        self.Q_eval.save_checkpoint()
    
    def learn(self):
        if self.memory.mem_idx < self.batch_size:
            return
        
        if self.learn_idx % self.replace_num == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
        
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        self.Q_eval.optimizer.zero_grad()

        if self.type == 'dqn':
            q_eval = self.Q_eval.forward(states)[batch_index, actions]
            q_next = self.Q_next.forward(next_states).max(dim=1)[0]
            
            q_next[dones] = 0.0
            
            q_target = rewards + self.gamma * q_next
            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        elif self.type == 'doubledqn':
            q_pred = self.Q_eval.forward(states)[batch_index, actions]
            q_next = self.Q_next.forward(next_states)
            q_eval = self.Q_eval.forward(next_states)
            
            max_actions = T.argmax(q_eval, dim=1)
            q_next[dones] = 0.0
            
            q_target = rewards + self.gamma * q_next[batch_index, max_actions]

            loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)

        loss.backward()
        self.Q_eval.optimizer.step()

        self.running_loss += loss.item()
        self.learn_idx += 1