import math

import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import regularizers, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from game.game import Action
from networks.network import UncertaintyAwareBaseNetwork

class EnsembleModel(Model):
  def __init__(self, models, selection_size) -> None:
      super(EnsembleModel, self).__init__()
      self.models = models
      self.selection_size = selection_size

  def call(self, input, train=False):
    outputs = []
    selected_models = random.sample(self.models, self.selection_size) if train else self.models
    for model in selected_models:
      outputs.append(model(input))

    prediction = tf.reduce_mean(outputs, axis=0)
    variance = tf.math.reduce_variance(outputs, axis=0)
    uncertainty_score = tf.reduce_mean(variance, axis=-1)
    return prediction, uncertainty_score

class EnsembleCartPoleNetwork(UncertaintyAwareBaseNetwork):

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 representation_size: int,
                 max_value: int,
                 num_dynamics_models: int,
                 selection_size: int,
                 hidden_neurons: int = 64,
                 weight_decay: float = 1e-4,
                 representation_activation: str = 'tanh'):
        self.state_size = state_size
        self.action_size = action_size
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1
        self.num_dynamics_models = num_dynamics_models
        self.selection_size = selection_size

        regularizer = regularizers.l2(weight_decay)
        representation_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                             Dense(representation_size, activation=representation_activation,
                                                   kernel_regularizer=regularizer)])
        value_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                    Dense(self.value_support_size, kernel_regularizer=regularizer)])
        policy_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                     Dense(action_size, kernel_regularizer=regularizer)])
        dynamic_network = self._build_dynamics_model(
          hidden_neurons=hidden_neurons,
          regularizer=regularizer,
          representation_activation=representation_activation,
          representation_size=representation_size)
        reward_network = Sequential([Dense(16, activation='relu', kernel_regularizer=regularizer),
                                     Dense(1, kernel_regularizer=regularizer)])

        super().__init__(representation_network, value_network, policy_network, dynamic_network, reward_network)

    def _value_transform(self, value_support: np.array) -> float:
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """

        value = self._softmax(value_support)
        value = np.dot(value, range(self.value_support_size))
        value = np.ndarray.item(value) ** 2
        return value

    def _reward_transform(self, reward: np.array) -> float:
        return np.ndarray.item(reward)

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        conditioned_hidden = np.concatenate((hidden_state, np.eye(self.action_size)[action.index]))
        return np.expand_dims(conditioned_hidden, axis=0)

    def _softmax(self, values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values))
        return values_exp / np.sum(values_exp)

    def _build_dynamics_model(self, hidden_neurons, regularizer, representation_size, representation_activation):
        networks = []
        for _ in range(self.num_dynamics_models):
          network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                      Dense(representation_size, activation=representation_activation,
                                            kernel_regularizer=regularizer)])
          networks.append(network)
        return EnsembleModel(networks, self.selection_size)
