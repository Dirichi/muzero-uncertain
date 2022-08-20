from typing import List
import operator
from functools import reduce

import gym
from gym.core import ObservationWrapper

from game.game import Action, AbstractGame
from game.gym_wrappers import ScalingObservationWrapper


class FlatImgObsWrapper(ObservationWrapper):
    """
    Encode the observed images into one flat array
    """

    def __init__(self, env):
        super().__init__(env, new_step_api=env.new_step_api)

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(imgSize,),
            dtype="uint8",
        )

    def observation(self, obs):
        image = obs["image"]
        return image.flatten()


class MiniGrid(AbstractGame):
    """The Gym CartPole environment"""

    def __init__(self, discount: float):
        super().__init__(discount)
        self.env = gym.make("MiniGrid-Empty-5x5-v0")
        self.env = FlatImgObsWrapper(self.env)
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
