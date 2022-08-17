import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from game.game import Action
from networks.network import BaseNetwork


class EnsembleCartPoleNetwork(BaseNetwork):

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 representation_size: int,
                 max_value: int,
                 num_dynamics_models: int,
                 hidden_neurons: int = 64,
                 weight_decay: float = 1e-4,
                 representation_activation: str = 'tanh'):
        self.state_size = state_size
        self.action_size = action_size
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1
        self.num_dynamics_models = num_dynamics_models

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
        value = np.asscalar(value) ** 2
        return value

    def _reward_transform(self, reward: np.array) -> float:
        return np.asscalar(reward)

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        conditioned_hidden = np.concatenate((hidden_state, np.eye(self.action_size)[action.index]))
        return np.expand_dims(conditioned_hidden, axis=0)

    def _softmax(self, values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values))
        return values_exp / np.sum(values_exp)

    def _build_dynamics_model(self, hidden_neurons, regularizer, representation_size, representation_activation):
        networks = []
        for i in range(self.num_dynamics_models):
          network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                      Dense(representation_size, activation=representation_activation,
                                            kernel_regularizer=regularizer)])
          networks.append(network)
        return EnsembleModel(networks)




class EnsembleModel(Model):
  def __init__(self, models) -> None:
      super(EnsembleModel, self).__init__()
      self.models = models

  def call(self, input):
    outputs = []
    for model in self.models:
      output = model(input)
      outputs.append(output)

    prediction = tf.reduce_mean(outputs, axis=0)
    uncertainty = tf.reduce_mean(tf.math.reduce_std(outputs, axis=0))
    return prediction, uncertainty
