from liir.src.envs.lbforaging.environment import ForagingEnv
from liir.src.envs.lbforaging.environment_subgoal import ForagingEnv as ForagingEnvSub
from liir.src.envs.lbforaging.environment_sparse import ForagingEnv as ForagingEnvSparse
# from .scenarios import load


def LBFEnv(args):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # # load scenario from script
    # scenario = load(args.scenario_name + ".py").Scenario()
    # # create world
    # world = scenario.make_world(args)
    # # create multiagent environment
    # env = MultiAgentEnv(world, scenario.reset_world,
    #                     scenario.reward, scenario.observation, scenario.info)

    if args.env_name == "LBF":
        env = ForagingEnv(players=args.num_agents,
                          max_player_level=2,
                          field_size=(6, 6),
                          max_food=3,
                          sight=1,
                          max_episode_steps=args.episode_length,
                          force_coop=True,
                          normalize_reward=True,)
    elif args.env_name == "LBF_sub":
        env = ForagingEnvSub(players=args.num_agents,
                             max_player_level=2,
                             field_size=(8, 8),
                             sight=1,
                             max_episode_steps=args.episode_length,
                             # force_coop=True,  # This is True by default in this env
                             normalize_reward=True, )
    elif args.env_name == "LBF_sparse":
        env = ForagingEnvSparse(players=args.num_agents,
                             max_player_level=3,
                             field_size=(6, 6),
                             max_food=3,
                             sight=1,
                             max_episode_steps=args.episode_length,
                             force_coop=False,
                             normalize_reward=True, )
    else:
        raise NotImplementedError

    return env