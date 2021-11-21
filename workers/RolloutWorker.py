import ray
import os
import gym
from ray.util import ActorPool

def env_creator(env_config):
    env = gym.make(env_config["env"])
    return env

@ray.remote
class RolloutWorker(object):
    def __init__(self,config):
        os.environ["MKL_NUM_THREADS"] = "1"
        self.env = config["env_creator"](config["env_config"])
        self.config = config

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        return self.env.step(action)


if __name__ == '__main__':
    ray.init()
    config = dict(env_config=dict(env="Breakout-v0"), env_creator=env_creator)
    # print(config)
    import time
    start = time.time()
    nums = 10
    workers = ActorPool([RolloutWorker.remote(config) for i in range(nums)])
    workers.map()
    print(ray.get(a))
    print(time.time() - start)
    # workers = ActorPool()
    # gen = ray.get([i.reset.remote() for i in workers])
    # print(gen)

