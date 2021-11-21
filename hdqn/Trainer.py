import numpy as np
from .Config import DefaultConfig
import torch
from copy import deepcopy
from .ConvNet import OptionCnnModel, critic_loss_fn, actor_loss_fn
from .utils import toTensor
from .ReplayBuffer import ReplayBuffer
from .logger import Logger

import time

class Trainer:
    def __init__(self, env, config):
        self.config = config
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
        self.model = OptionCnnModel(
            in_features=env.observation_space.shape[2],
            num_actions=env.action_space.n,
            num_options=config.num_options,
            temperature=config.temp,
            eps_start=config.epsilon_start,
            eps_min=config.epsilon_min,
            eps_decay=config.epsilon_decay,
            eps_test=config.optimal_eps,
            device=self.device
        )
        self.target_model = deepcopy(self.model)
        self.model = self.model.to(self.device)
        self.target_model = self.target_model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(capacity=config.max_history, seed = config.seed)
        self.logger = Logger(logdir=config.logdir, run_name=f"{OptionCnnModel.__name__}-{self.config.env}-{config.exp_name}-{time.ctime()}")

    def train(self):
        steps = 0
        self.model.train()
        while steps < self.config.max_steps_total:
            if self.logger.n_eps % self.config.model_save_freq == 0 and self.logger.n_eps > 0:
                torch.save({"model_params": self.model.state_dict()},
                           f"models/{OptionCnnModel.__name__}-{self.config.env}-{self.config.exp_name}-{time.ctime()}.pth")
                print("*" * 20 + "Saving the Model" + "*" * 20)
            steps = self.train_one_episode(self.env, steps, self.model, self.target_model, self.config, self.buffer, self.optim, self.logger, self.device)

    def train_one_episode(self,
                          env,
                          steps,
                          model: OptionCnnModel,
                          target_model: OptionCnnModel,
                          config: DefaultConfig,
                          buffer: ReplayBuffer,
                          optim: torch.optim.Adam,
                          logger: Logger,
                          device):
        rewards = 0
        option_lengths = {opt: [] for opt in range(config.num_options)}
        obs = env.reset()
        obs = toTensor(obs).to(device)
        state = model.get_state(obs)
        greedy_option = model.greedy_option(state)
        current_option = 0
        done = False
        ep_steps = 0
        option_termination = True
        curr_op_len = 0
        while not done and ep_steps < config.max_steps_ep:
            epsilon = model.epsilon
            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(config.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0
            action, logp, entropy = model.get_action(state, [current_option])
            next_obs, reward, done, _ = env.step(action)
            next_obs = toTensor(next_obs).to(device)
            buffer.push(np.array(obs[0]), current_option, reward, action, logp, entropy, np.array(next_obs[0]), done)
            state = model.get_state(next_obs)
            option_termination, greedy_option = model.predict_option_termination(state, current_option)
            rewards += reward
            actor_loss, critic_loss = None, None
            if len(buffer) > config.batch_size:
                if steps % config.update_frequency == 0 and steps > 0:
                    data_batch = buffer.last(config.batch_size, device, config.update_frequency)
                    actor_loss = actor_loss_fn(data_batch, model, target_model, config)
                    data_batch = buffer.sample(config.batch_size, device)
                    critic_loss = critic_loss_fn(data_batch, model, target_model, config)
                    loss = actor_loss + critic_loss
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                if steps % config.freeze_interval == 0:
                    target_model.load_state_dict(model.state_dict())
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs
            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)
        logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)
        return steps



