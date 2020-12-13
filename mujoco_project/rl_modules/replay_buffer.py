import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""


class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func, load_path):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        }
        if load_path != '':
            self._preload_buffer(load_path)
            print("Pre-loaded buffer successfully from {}".format(load_path))
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def save_buffer(self, path: str):
        """
        Method to save the buffer
        :params: path - string path of .npy file to save the buffer to
        """
        np.save(path, self.buffers)

    def _preload_buffer(self, path: str):
        """
        Method to load the buffer from given npy file and save 50%
        :params: path - string path of .npy file to load buffer from
        """
        # saved_obj is numpy array with only one dictionary
        saved_obj = np.load(path)
        # copy complete buffers dictionary
        self.buffers = saved_obj[()]
        buff_size, times, obs_size = self.buffers['obs'].shape
        # cut 50% of the last. Even though the filling occurs randomly once the buffer reaches capacity
        half_buff = int(buff_size / 2)

        # extract the final half buffer
        rep_obs = np.copy(self.buffers['obs'][half_buff:, :, :])
        rep_ag = np.copy(self.buffers['ag'][half_buff:, :, :])
        rep_g = np.copy(self.buffers['g'][half_buff:, :, :])
        rep_act = np.copy(self.buffers['actions'][half_buff:, :, :])

        # replace the first half of the current buffer with those
        n_copy = rep_obs.shape[0]
        self.buffers['obs'][:n_copy, :, :] = np.copy(rep_obs)
        self.buffers['ag'][:n_copy, :, :] = np.copy(rep_ag)
        self.buffers['g'][:n_copy, :, :] = np.copy(rep_g)
        self.buffers['actions'][:n_copy, :, :] = np.copy(rep_act)
        # set the index used in accessing the buffer
        # initially, number of transitions stored equals the current size of the buffer
        self.current_size = min(n_copy, self.size)
        self.n_transitions_stored = n_copy
