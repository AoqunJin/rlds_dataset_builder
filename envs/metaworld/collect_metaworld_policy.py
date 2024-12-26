import metaworld.envs.mujoco.env_dict as _env_dict
import random
import numpy as np
import tqdm
import os

from mw_tools import POLICIES, setup_metaworld_env

# Parameters
N_TRAIN_EPISODES = 20
# N_TEST_EPISODES = 20
EPISODE_LENGTH = 500

# Data directories
TRAIN_DIR = 'data/train'
# TEST_DIR = 'data/test'
os.makedirs(TRAIN_DIR, exist_ok=True)
# os.makedirs(TEST_DIR, exist_ok=True)

# Initialize Metaworld environments
benchmark = _env_dict.ML10_V2  # Construct the benchmark

def collect_episode(env, policy, instruction, episode_length, path):
    """Collect data for a single episode and save it as an .npy file."""
    episode = []
    obs, info = env.reset()
    
    for step in range(episode_length):
        # action = env.action_space.sample()  # Sample a random action
        action = np.clip(policy.get_action(obs), -1, 1)
        next_obs, reward, done, truncate, info = env.step(action)

        # Save step data
        episode.append({
            'image': np.asarray(env.render(), dtype=np.uint8),  # Image uint8
            'wrist_image': np.asarray(np.random.rand(256, 256, 3) * 255, dtype=np.uint8),  # Placeholder wrist image
            'state': np.asarray(obs, dtype=np.float32),  # State from the environment
            'action': np.asarray(action, dtype=np.float32),  # Action taken
            # 'reward': np.asarray(reward, dtype=np.float32),  # Reward received
            'language_instruction': instruction,  # Dummy language instruction
        })

        obs = next_obs
        if info['success']: break

    np.save(path, episode)

def collect_data(envs, env_names, num_episodes, save_dir, episode_length):
    """Collect and save data for multiple episodes."""
    for i, env in enumerate(tqdm.tqdm(envs, desc=f"Collecting data in {save_dir}")):
        env.max_path_length = EPISODE_LENGTH
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        policy = POLICIES[env_names[i]]()
        instruction = " ".join(env_names[i].split('-')[:-1])
        for episode_idx in range(num_episodes):
            episode_path = os.path.join(save_dir, f'episode_env{i}_ep{episode_idx}.npy')
            collect_episode(env, policy, instruction, episode_length, episode_path)

# Training environments
training_envs = []
training_env_names = []
for name in benchmark['train'].keys():
    env = setup_metaworld_env(name + '-goal-observable')    
    training_envs.append(env)
    training_env_names.append(name)

# Testing environments
# testing_envs = []
# testing_env_names = []
# for name, env_cls in benchmark['test'].items():
#     env = env_cls()
#     task = random.choice([task for task in benchmark.test_tasks if task.env_name == name])
#     env.set_task(task)
#     testing_envs.append(env)
# testing_env_names.append(name)

# Collect training data
collect_data(training_envs, training_env_names, N_TRAIN_EPISODES, TRAIN_DIR, EPISODE_LENGTH)

# Collect testing data
# collect_data(testing_envs, testing_env_names, N_TEST_EPISODES, TEST_DIR, EPISODE_LENGTH)

print('Data collection complete!')
