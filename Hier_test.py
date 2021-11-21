from hdqn.Trainer import Trainer
from hdqn.Config import DefaultConfig
import gym
from ray.rllib.env.atari_wrappers import wrap_deepmind
if __name__ == '__main__':
    config = DefaultConfig()
    config.env = "BreakoutNoFrameskip-v0"
    env = gym.make(config.env)
    env = wrap_deepmind(env)
    trainer = Trainer(env, config)
    trainer.train()