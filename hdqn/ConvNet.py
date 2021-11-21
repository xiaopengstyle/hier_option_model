import tensorboard
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
from math import exp



class OptionCnnModel(nn.Module):
    def __init__(self,
                 in_features,
                 num_actions,
                 num_options,
                 temperature=1.0,
                 eps_start=1.0,
                 eps_min=0.1,
                 eps_decay=int(1e6),
                 eps_test=0.05,
                 device="cpu",
                 testing=False,
                 activation=nn.Tanh):
        super(OptionCnnModel, self).__init__()
        self.in_channels = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.magic_number = 7 * 7 * 64
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.num_steps = 0

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            activation(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            activation(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            activation(),
            nn.modules.Flatten(),
            nn.Linear(self.magic_number, 512),
            activation()
        )
        self.Q = nn.Linear(512, num_options)
        self.terminations = nn.Linear(512, num_options)
        self.options_W = nn.Parameter(torch.zeros(num_options, 512, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

    def get_state(self, obs):
        return self.features(obs)

    def get_Q(self, state):
        return self.Q(state)

    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        next_option = self.get_Q(state).argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()

    def get_terminations(self, state):
        return self.terminations(state).sigmoid()

    def get_action(self, state, option):
        logits = state @ self.options_W[option] + self.options_b[option]
        action_dist = logits / self.temperature
        action_dist = action_dist.softmax(dim=-1)
        action_dist = Categorical(action_dist)
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action.item(), logp, entropy

    def greedy_option(self, state):
        return self.get_Q(state).argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps = self.num_steps + 1
        else:
            eps = self.eps_test
        return eps


def critic_loss_fn(data_batch, model: OptionCnnModel, target_model, config):
    obs, options, rewards, actions, logp, entropy, next_obs, dones = data_batch
    batch_index = torch.arange(len(options)).long()
    # 计算当前时刻的Q
    states = model.get_state(obs)
    Q = model.get_Q(states)
    # Q learning has target
    next_states_target = target_model.get_state(next_obs)
    next_Q_target = target_model.get_Q(next_states_target.detach())
    # termination probability
    next_states = model.get_state(next_obs)
    next_termination_probs = model.get_terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_index, options]
    gt = rewards + (1 - dones) * config.gamma *(
        (1 - next_options_term_prob) * next_Q_target[batch_index, options] +
        next_options_term_prob * next_Q_target.max(dim=-1)[0])
    return (Q[batch_index,options] - gt.detach()).pow(2).mean()


def actor_loss_fn(data_batch, model: OptionCnnModel, target_model, config):
    obs, options, rewards, action, logp, entropy, next_obs, dones = data_batch
    batch_index = torch.arange(len(options)).long()
    # 计算当前时刻的Q
    states = model.get_state(obs)
    next_states = model.get_state(next_obs)
    next_states_target = target_model.get_state(next_obs)
    option_term_prob = model.get_terminations(states)[batch_index, options]
    next_option_termination_probs = model.get_terminations(next_states)[batch_index, options].detach()

    Q = model.get_Q(states).detach().squeeze()
    next_Q_target = target_model.get_Q(next_states_target).detach().squeeze()

    gt = rewards + (1 - dones) * config.gamma * (
        (1 - next_option_termination_probs) * next_Q_target[batch_index,options] +
        next_option_termination_probs * next_Q_target.max(dim=-1)[0])

    term_loss = option_term_prob * (Q[batch_index, options].detach() - Q.max(dim=-1)[0].detach() + config.termination_reg) * (1 - dones)
    policy_loss = -logp * (gt.detach() - Q[batch_index, options]) - config.entropy_reg * entropy
    loss = term_loss + policy_loss
    return loss.mean()




