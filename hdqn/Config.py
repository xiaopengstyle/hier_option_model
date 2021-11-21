

class DefaultConfig:
    env = "PongFrameskip-v0"
    optimal_eps = 0.05
    frame_skip = 4
    lr = 0.00025
    gamma = 0.99
    # 探索相关的设置
    epsilon_start = 1.0
    epsilon_min = 0.1
    epsilon_decay = int(1e6)
    # 训练相关
    max_history = int(1e6)
    batch_size = 32
    # target的更新频率
    freeze_interval = int(1e4)
    # 在计算前收集的action
    update_frequency = int(4)
    # 常用设置
    termination_reg = 0.01
    entropy_reg = 0.01
    num_options = 8
    # 温度系数
    temp = 1
    # 限制可能时长太长
    max_steps_ep = int(1.8e4)
    # 时长
    max_steps_total = int(4e6)
    # 其他参数
    cuda = True
    seed = 0
    logdir = "runs"
    exp_name = None
    switch_goal = False
    # model_save
    model_save_freq = int(1e7)