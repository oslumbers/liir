import logging
from collections import namedtuple, defaultdict
import random
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np
from envs.lbforaging.agent import Agent
# from onpolicy.envs.lb_foraging.lbforaging.agents.heuristic_agent_sparse import H1, H2, H3, H4


class HeuristicAgent(Agent):

    name = "Enemy"

    def _center_of_players(self, players):
        coords = np.array([player.position for player in players])
        return np.rint(coords.mean(axis=0))

    def _move_towards(self, target, allowed):

        y, x = self.observed_position
        r, c = target

        if r < y and Action.NORTH in allowed:
            return Action.NORTH
        elif r > y and Action.SOUTH in allowed:
            return Action.SOUTH
        elif c > x and Action.EAST in allowed:
            return Action.EAST
        elif c < x and Action.WEST in allowed:
            return Action.WEST
        else:
            raise ValueError("No simple path found")

    def step(self, obs):
        raise NotImplemented("Heuristic agent is implemented by H1-H4")

class H1(HeuristicAgent):
    """
	H1 agent always goes to the closest food
	"""

    name = "Enemy"

    def step(self, obs):
        try:
            r, c = self._closest_food(obs)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H2(HeuristicAgent):
    """
	H2 Agent goes to the one visible food which is closest to the centre of visible players
	"""

    name = "Enemy"

    def step(self, obs):
        
        players = [player for player in obs.players if player.battle_lines == 'Enemy']
        players_center = self._center_of_players(players)

        try:
            r, c = self._closest_food(obs, None, players_center)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H3(HeuristicAgent):
    """
	H3 Agent always goes to the closest food with compatible level
	"""

    name = "Enemy"

    def step(self, obs):

        try:
            r, c = self._closest_food(obs, self.level)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H4(HeuristicAgent):
    """
	H4 Agent goes to the one visible food which is closest to all visible players
	 such that the sum of their and H4's level is sufficient to load the food
	"""

    name = "Enemy"

    def step(self, obs):
        
        players = [player for player in obs.players if player.battle_lines == 'Enemy']
        players_center = self._center_of_players(players)
        # print (f'This is the observed players center: {players_center}')
        players_sum_level = sum([a.level for a in players])

        try:
            r, c = self._closest_food(obs, players_sum_level, players_center)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Agent"

# TODO: JIANHONG
'''
1. Change the observation space, for which only the agents that controlled by MARL can be observed.
2. Change the action space, ~~
3. Change the reward function
'''
class ForagingEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    # TODO: JIANHONG
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self", "battle_lines"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        max_player_level,
        field_size,
        max_food,
        sight,
        max_episode_steps,
        force_coop,
        normalize_reward=True,
    ):
        self.logger = logging.getLogger(__name__)
        self.seed()

        # TODO: SET UP ENEMIES AND AGENTS
        # self.players = [Player() for _ in range(players)]
        agents = [Player() for _ in range(players)]
        enemies = [Player() for _ in range(players)]
        for e in enemies:
            e.set_controller(H4(player=e))
        self.players = agents + enemies
        self.enemy_actions = [0] * len(enemies)

        self.field = np.zeros(field_size, np.int32)

        self.max_food = max_food
        self._food_spawned = 0.0
        self.max_player_level = max_player_level
        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None
        self.n_agents = len(agents)

        # TODO: JIANHONG    
        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(6)] * len(agents)))
        self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()] * len(agents)))
        self.share_observation_space = gym.spaces.Tuple(tuple([self._get_shared_observation_space()] * len(agents)))

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward

        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        field_x = self.field.shape[1]
        field_y = self.field.shape[0]
        # field_size = field_x * field_y

        max_food = self.max_food
        max_food_level = self.max_player_level * len(self.players)

        min_obs = [-1, -1, 0] * max_food + [0, 0, 1] * len(self.players)
        max_obs = [field_x, field_y, max_food_level] * max_food + [
            field_x,
            field_y,
            self.max_player_level,
        ] * len(self.players)

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    def _get_shared_observation_space(self):
        """The Observation Space for each agent.
        for n_players:
            - all of the board (board_size^2) with foods
            - player description (x, y, level)*player_count
        """

        shared_obs_space_min = self.observation_space[0].low
        shared_obs_space_high = self.observation_space[0].high
        # TODO: JIANHONG
        # for obs_space in self.observation_space[1:]:
        for obs_space in self.observation_space[1:self.n_agents]:
            shared_obs_space_min = np.append(shared_obs_space_min, obs_space.low)
            shared_obs_space_high = np.append(shared_obs_space_high, obs_space.high)

        return gym.spaces.Box(shared_obs_space_min, shared_obs_space_high, dtype=np.float32)


    @classmethod
    def from_obs(cls, obs):

        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]
    
    def adjacent_agents(self, row, col):
        return [
            player
            for player in self.players[:self.n_agents]
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def adjacent_enemies(self, row, col):
        return [
            player
            for player in self.players[self.n_agents:]
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_food(self, max_food, max_level):

        food_count = 0
        attempts = 0
        min_level = max_level if self.force_coop else 1

        while food_count < max_food and attempts < 1000:
            attempts += 1
            row = self.np_random.randint(1, self.rows - 1)
            col = self.np_random.randint(1, self.cols - 1)

            # check if it has neighbors:
            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue

            self.field[row, col] = (
                min_level
                if min_level == max_level
                else self.np_random.randint(min_level, max_level)
            )
            food_count += 1
        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):

        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self, max_player_level):

        for player in self.players:

            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.randint(0, self.rows - 1)
                col = self.np_random.randint(0, self.cols - 1)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.np_random.randint(1, max_player_level),
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                    battle_lines=a.name
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    # TODO: the target function to be modified
    def _make_gym_obs(self, observations):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self and (p.battle_lines == 'Agent')
            ] + [
                p for p in observation.players if not p.is_self and (p.battle_lines == 'Enemy')
            ]

            for i in range(self.max_food):
                obs[3 * i] = -1
                obs[3 * i + 1] = -1
                obs[3 * i + 2] = 0

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[3 * i] = y
                obs[3 * i + 1] = x
                obs[3 * i + 2] = observation.field[y, x]

            for i in range(len(self.players)):
                obs[self.max_food * 3 + 3 * i] = -1
                obs[self.max_food * 3 + 3 * i + 1] = -1
                obs[self.max_food * 3 + 3 * i + 2] = 0

            for i, p in enumerate(seen_players):
                obs[self.max_food * 3 + 3 * i] = p.position[0]
                obs[self.max_food * 3 + 3 * i + 1] = p.position[1]
                obs[self.max_food * 3 + 3 * i + 2] = p.level

            return obs

        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward

        nobs = tuple([make_obs_array(obs) for obs in observations])
        nreward = [[get_player_reward(obs)] for obs in observations]
        ndone = [obs.game_over for obs in observations]
        # ninfo = [{'observation': obs} for obs in observations]
        ninfo = {}

        return nobs, nreward, ndone, ninfo

    def reset(self):
        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players(self.max_player_level)
        player_levels = sorted([player.level for player in self.players])

        self.spawn_food(
            self.max_food, max_level=sum(player_levels[:3])
        )
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()
        self.agent_sum_rewards = 0
        self.enemy_sum_rewards = 0
        # observations = [self._make_obs(player) for player in self.players]
        # nobs, nreward, ndone, ninfo = self._make_gym_obs(observations)
        
        # return nobs
        # TODO: JIANHONG
        # return nobs[:self.n_agents]
        observations = [self._make_obs(player) for player in self.players]
        observations_agents = observations[:self.n_agents]
        observations_enemies = observations[self.n_agents:]
        nobs, nreward, ndone, ninfo = self._make_gym_obs(observations_agents)

        enemies = self.players[self.n_agents:]
        self.enemy_actions = []
        for i in range(len(observations_enemies)):
            action = enemies[i].step(observations_enemies[i])
            # print (f'This is the action from heuristic agents: {tuple(action)}')
            self.enemy_actions.append(action)

        return nobs

    def loading_players(self, player_set, battle_lines):
        while player_set:
            # find adjacent food
            player = player_set.pop()
            # TODO: JIANHONG: some food may be captured by the opponents due to the chance given by coin
            try:
                frow, fcol = self.adjacent_food_location(*player.position)
            except Exception:
                continue
            food = self.field[frow, fcol]

            if battle_lines == 'Agent':
                adj_players = self.adjacent_agents(frow, fcol)
            elif battle_lines == 'Enemy':
                adj_players = self.adjacent_enemies(frow, fcol)
            adj_players = [
                p for p in adj_players if p in player_set or p is player
            ]

            adj_player_level = sum([a.level for a in adj_players])

            player_set = player_set - set(adj_players)

            if adj_player_level < food:
                # failed to load
                continue

            # else the food was loaded and each player scores points
            for a in adj_players:
                reward = float(a.level * food)
                if self._normalize_reward:
                    reward = reward / float(
                        adj_player_level * self._food_spawned
                    )
                if battle_lines == 'Agent':
                    self.agent_sum_rewards += reward
                elif battle_lines == 'Enemy':
                    self.enemy_sum_rewards += reward

            # and the food is removed
            self.field[frow, fcol] = 0

    def step(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = 0

        # TODO: JIANHONG
        # check the validity of actions (for both enemies and agents)
        actions = [Action(a) for a in actions]

        enemy_actions = self.enemy_actions

        actions.extend(enemy_actions)

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        # TODO: JIANHONG
        loading_agents = set()
        loading_enemies = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # TODO: JIANHONG
        # so check for collisions

        for player, action in zip(self.players[:self.n_agents], actions[:self.n_agents]):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_agents.add(player)

        for player, action in zip(self.players[self.n_agents:], actions[self.n_agents:]):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_enemies.add(player)

        # and do movements for non colliding players
        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        # TODO: JIANHONG
        # finally process the loadings:
        coin = np.random.rand()
        if coin > 0.5:
            self.loading_players(loading_agents, battle_lines='Agent')
            self.loading_players(loading_enemies, battle_lines='Enemy')
        else:
            self.loading_players(loading_enemies, battle_lines='Enemy')
            self.loading_players(loading_agents, battle_lines='Agent')

        self._game_over = (
            self.field.sum() == 0 or self._max_episode_steps <= self.current_step
        )

        self._gen_valid_moves()

        # TODO: JIANHONG
        # sparse reward: no any immediate reward
        # if self._game_over and (self.agent_sum_rewards > self.enemy_sum_rewards):
        #     for agent in self.players[:self.n_agents]:
        #         agent.reward = self.agent_sum_rewards - self.enemy_sum_rewards
        # elif self._game_over and (self.agent_sum_rewards <= self.enemy_sum_rewards):
        #     for agent in self.players[:self.n_agents]:
        #         agent.reward = self.agent_sum_rewards - self.enemy_sum_rewards
        if self._game_over:
            for agent in self.players[:self.n_agents]:
                agent.reward = self.agent_sum_rewards - self.enemy_sum_rewards
        else:
            for agent in self.players[:self.n_agents]:
                agent.reward = 0

        for p in self.players:
            p.score += p.reward

        # TODO: JIANHONG
        observations = [self._make_obs(player) for player in self.players]
        observations_agents = observations[:self.n_agents]
        observations_enemies = observations[self.n_agents:]

        info_gym_agents = self._make_gym_obs(observations_agents)

        enemies = self.players[self.n_agents:]
        self.enemy_actions = []
        for i in range(len(observations_enemies)):
            action = enemies[i].step(observations_enemies[i])
            self.enemy_actions.append(action)

        return info_gym_agents

    def _init_render(self):
        from .rendering_sparse import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()