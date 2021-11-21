import gym
import gym_minigrid

if __name__ == '__main__':
    env_name = "MiniGrid-FourRooms-v0"
    env = gym.make(env_name)
    print(env.reset())