import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4


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
            return "Player"


class ForagingEnvSuboptimal(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        field_size,
        sight,
        max_episode_steps,
        total_suboptimal_reward=2,
        optimal_reward=1.5,
    ):

        assert total_suboptimal_reward/2 < optimal_reward <  total_suboptimal_reward, "total_suboptimal_reward/2 < optimal_reward <  total_suboptimal_reward"
        self.logger = logging.getLogger(__name__)
        self.seed()
        self.players = [Player() for _ in range(2)]

        self.field = np.zeros(field_size, np.int32)

        self.neutral_area_height = None
        self.r_area_height = None
        self.divide_field()
        self.r = total_suboptimal_reward
        self.R = optimal_reward
        self.optimal_food_loc = None
        self.optimal_found = False
        self.total_suboptimal_food = self.field_size[1] * self.r_area_height + 1
        self.max_food = (1+2*sight)**2

        self.sight = sight
        self._game_over = None

        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(5)] * len(self.players)))
        self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()] * len(self.players)))
        self.share_observation_space = gym.spaces.Tuple(tuple([self._get_shared_observation_space()] * len(self.players)))

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self.viewer = None

        self.n_agents = len(self.players)


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
        max_food_level = self.R

        min_obs = [-1, -1, 0] * max_food + [0, 0] * len(self.players)
        max_obs = [field_x, field_y, max_food_level] * max_food + [field_x, field_y] * len(self.players)

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    def _get_shared_observation_space(self):
        """The Observation Space for each agent.
        for n_players:
            - all of the board (board_size^2) with foods
            - player description (x, y, level)*player_count
        """

        shared_obs_space_min = self.observation_space[0].low
        shared_obs_space_high = self.observation_space[0].high
        for obs_space in self.observation_space[1:]:
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

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_suboptimal_food(self):
        for col in range(self.field_size[1]):
            for row in range(self.r_area_height):
                self.field[row, col] = self.r / self.total_suboptimal_food

    def _is_empty_location(self, row, col):

        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players_in_neutral(self):

        for player in self.players:

            attempts = 0
            player.reward = 0
            rows = (self.r_area_height, self.r_area_height+self.neutral_area_height)
            while attempts < 1000:
                row = self.np_random.randint(*rows)
                col = self.np_random.randint(0, self.cols - 1)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.r+1,
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
                # and self.field[player.position[0] - 1, player.position[1]] == 0
                and player.position[0] != (self.r_area_height+self.neutral_area_height)
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                # and self.field[player.position[0] + 1, player.position[1]] == 0
                and player.position[0] != (self.r_area_height-1)
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                # and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                # and self.field[player.position[0], player.position[1] + 1] == 0
            )

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

    def _make_gym_obs(self, observations):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            for i in range(self.max_food):
                obs[3 * i] = -1
                obs[3 * i + 1] = -1
                obs[3 * i + 2] = 0

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[3 * i] = y
                obs[3 * i + 1] = x
                obs[3 * i + 2] = observation.field[y, x]

            player_position = [p.position for p in observation.players if p.is_self][0]
            optimal_on_sight = (min(self._transform_to_neighborhood(player_position, self.sight, self.optimal_food_loc))>=0) and \
                               (max(self._transform_to_neighborhood(player_position, self.sight, self.optimal_food_loc))<=2*self.sight)
            if optimal_on_sight and self.get_relative_position()=='coop':
                optimal_y, optimal_x = self._transform_to_neighborhood(player_position, self.sight, self.optimal_food_loc)
                obs[(self.max_food-1) * 3] = optimal_y
                obs[(self.max_food-1) * 3 + 1] = optimal_x
                obs[(self.max_food-1) * 3 + 2] = self.R

            for i in range(len(self.players)):
                obs[self.max_food * 3 + 2 * i] = -1
                obs[self.max_food * 3 + 2 * i + 1] = -1

            for i, p in enumerate(seen_players):
                obs[self.max_food * 3 + 2 * i] = p.position[0]
                obs[self.max_food * 3 + 2 * i + 1] = p.position[1]

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

    def divide_field(self):
        assert self.field_size[0]==self.field_size[1]
        assert self.field_size[0]>2
        h = self.field_size[1]
        if h % 3 == 0:
            neutral_area_height = h // 3
            r_area_height = h // 3
        elif h % 3 == 1:
            neutral_area_height = h // 3 + 1
            r_area_height = h // 3
        else:  # h%3==2:
            neutral_area_height = h // 3
            r_area_height = h // 3 + 1
        self.neutral_area_height = neutral_area_height
        self.r_area_height = r_area_height
        assert (self.neutral_area_height + 2*self.r_area_height) == self.field_size[0]

    def get_area(self, row):
        assert row < 2*self.r_area_height+self.neutral_area_height, "row too large!"
        if row < self.r_area_height:
            return 'top'
        if row < (self.r_area_height+self.neutral_area_height):
            return 'neutral'
        if row < (2*self.r_area_height+self.neutral_area_height):
            return 'bottom'

    def get_relative_position(self):
        areas_players = [self.get_area(p.position[0]) for p in self.players]
        if areas_players[0]==areas_players[1] and areas_players[0]=='top':
            return 'compete'
        elif areas_players[0]==areas_players[1] and areas_players[0]=='bottom':
            return 'coop'
        elif areas_players[0]!=areas_players[1] and 'bottom' in areas_players and 'top' in areas_players:
            return 'antagonic'
        else:
            return 'neutral'

    def reset(self):
        self.field = np.zeros(self.field_size)
        self.spawn_suboptimal_food()
        self.spawn_players_in_neutral()
        self.optimal_food_loc = (self.field_size[0]-1, self.field_size[1]-1)

        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        observations = [self._make_obs(player) for player in self.players]
        nobs, nreward, ndone, ninfo = self._make_gym_obs(observations)
        return nobs

    def step(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
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

        # and do movements for non colliding players

        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        #check relative position of players
        areas_players = [self.get_area(p.position[0]) for p in self.players]
        if areas_players[0]==areas_players[1] and areas_players[0]=='top':
            game = 'compete'
        elif areas_players[0]==areas_players[1] and areas_players[0]=='bottom':
            game = 'coop'
        elif areas_players[0]!=areas_players[1] and 'bottom' in areas_players and 'top' in areas_players:
            game = 'antagonic'
        else:
            game = 'neutral'

        # finally process the loadings:
        if game in ['antagonic']:
            player0 = self.players[0]
            player1 = self.players[1]
            if self.get_area(player0.position[0]) == 'top':
                frow, fcol = player0.position[0], player0.position[1]
                player0.reward = self.field[frow, fcol]
                player1.reward = -self.field[frow, fcol]
            else:
                frow, fcol = player1.position[0], player1.position[1]
                player0.reward = -self.field[frow, fcol]
                player1.reward = self.field[frow, fcol]
            self.field[frow, fcol] = 0  # and the food is removed

        elif game in ['compete', 'neutral']:
            for player in self.players:
                # Regular loading
                frow, fcol = player.position[0], player.position[1]
                player.reward = self.field[frow, fcol]
                self.field[frow, fcol] = 0  # and the food is removed
        else:
            if not self.optimal_found:
                for player in self.players:
                    frow, fcol = player.position[0], player.position[1]
                    if frow == self.optimal_food_loc[0] and fcol == self.optimal_food_loc[1]:
                        self.players[0].reward = self.R
                        self.players[1].reward = self.R
                        self.optimal_found = True
                        self.field = np.zeros_like(self.field)  # And game ends
                        break

        self._game_over = (
            self.field.sum() < 1e-6 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        for p in self.players:
            p.score += p.reward

        observations = [self._make_obs(player) for player in self.players]
        return self._make_gym_obs(observations)

    def _init_render(self):
        from .rendering_suboptimal import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()