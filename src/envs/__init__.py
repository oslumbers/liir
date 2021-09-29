from functools import partial
from envs.multiagentenv import MultiAgentEnv
from envs.lbforaging.environment import ForagingEnv
from envs.lbforaging.environment_subgoal import ForagingEnv as ForagingEnvSubGoal
from envs.lbforaging.environment_sparse import ForagingEnv as ForagingEnvSparse
from envs.lbforaging.environment_suboptimal import ForagingEnvSuboptimal 
import sys
import os
import gym
from gym import ObservationWrapper, spaces
from gym.spaces import flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}

class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done)
            done = len(observation) * [True]
        return observation, reward, done, info
        

class FlattenObservation(ObservationWrapper):
    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )

class _GymmaWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, **kwargs):
        self.episode_limit = time_limit
        if key == 'LBF4x4':
            self._env = ForagingEnv(players=2,
                          max_player_level=2,
                          field_size=(4, 4),
                          max_food=3,
                          sight=1,
                          max_episode_steps=time_limit,
                          force_coop=True,
                          normalize_reward=True,)
        elif key == 'LBF6x6':
            self._env = ForagingEnv(players=2,
                          max_player_level=2,
                          field_size=(6, 6),
                          max_food=3,
                          sight=1,
                          max_episode_steps=time_limit,
                          force_coop=True,
                          normalize_reward=True,)
        elif key == 'LBF_subgoal':
            self._env = ForagingEnvSubGoal(players=2,
                             max_player_level=2,
                             field_size=(8, 8),
                             sight=1,
                             max_episode_steps=time_limit,
                             # force_coop=True,  # This is True by default in this env
                             normalize_reward=True, )
        elif key == 'LBF_sparse':
            self._env = ForagingEnvSparse(players=2,
                             max_player_level=3,
                             field_size=(6, 6),
                             max_food=3,
                             sight=1,
                             max_episode_steps=time_limit,
                             force_coop=False,
                             normalize_reward=True, )
        elif key == 'LBF_suboptimal':
            self._env = ForagingEnvSuboptimal(players=2,
                            max_player_level=2,
                            field_size=(6, 6),
                            sight=1,
                            max_episode_steps=time_limit,
                            force_coop=False,
                            normalize_reward=True, )

        #self._env = TimeLimit(gym.make(f"{key}"), max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        self.n_agents = self._env.n_agents
        self._obs = None
        
        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)

    def step(self, actions):
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self._env.step(actions)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        
        return float(sum(reward[0])), all(done), {}

    def get_obs(self):
        return self._obs
    
    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid
    
    def get_total_actions(self):
        return flatdim(self.longest_action_space)

    def reset(self):
        self._obs = self._env.reset()
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)