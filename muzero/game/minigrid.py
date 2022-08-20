from typing import List

import gym
import gym_minigrid
from gym.core import ObservationWrapper

from game.game import Action, AbstractGame
from game.gym_wrappers import ScalingObservationWrapper


# Copied from gym_minigrid because of an issue with gym_minigrd.wrappers
# Can be swapped out for `from gym_minigrid.wrappers import ImgObsWrapper`
# when the gym_minigrid issue is fixed.
class ImgObsWrapper(ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env, new_step_api=env.new_step_api)
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, obs):
        return obs["image"]


class MiniGrid(AbstractGame):
    """The Gym CartPole environment"""

    def __init__(self, discount: float):
        super().__init__(discount)
        self.env = gym.make("MiniGrid-Empty-5x5-v0")
        self.env = ImgObsWrapper(self.env)
        self.env = ScalingObservationWrapper(self.env)
        self.actions = list(map(lambda i: Action(i), range(self.env.action_space.n)))
        self.observations = [self.env.reset()]
        self.done = False

    @property
    def action_space_size(self) -> int:
        """Return the size of the action space."""
        return len(self.actions)

    def step(self, action) -> int:
        """Execute one step of the game conditioned by the given action."""

        observation, reward, done, _ = self.env.step(action.index)
        self.observations += [observation]
        self.done = done
        return reward

    def terminal(self) -> bool:
        """Is the game is finished?"""
        return self.done

    def legal_actions(self) -> List[Action]:
        """Return the legal actions available at this instant."""
        return self.actions

    def make_image(self, state_index: int):
        """Compute the state of the game."""
        return self.observations[state_index]
