"""Training module: this is where MuZero neurons are trained."""

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MSE
from typing import List

from config import MuZeroConfig
from networks.network import BaseNetwork
from networks.shared_storage import SharedStorage
from training.replay_buffer import ReplayBuffer
from tensorflow.keras.models import Model

class Accumulator:
    def __init__(self) -> None:
        self.total = 0
        self.count = 0

    def add(self, values):
        self.total += sum(values)
        self.count += len(values)

    def average(self):
        return self.total / self.count


def train_ensemble_aware_network(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, epochs: int):
    network = storage.current_network
    optimizer = storage.optimizer
    accumulator = Accumulator()

    for _ in range(epochs):
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        update_ensemble_dynamics_model(config, optimizer, network, batch)
        update_weights(config, optimizer, network, accumulator, batch)
        storage.save_network(network.training_steps, network)

    return accumulator.average()

def scale_gradient(tensor, scale: float):
    """Trick function to scale the gradient in tensorflow"""
    return (1. - scale) * tf.stop_gradient(tensor) + scale * tensor

def update_ensemble_dynamics_model(config: MuZeroConfig, optimizer: tf.keras.optimizers, network: BaseNetwork, batch):
    # for model in network.dynamic_network.models:
    #     # Call once to initialize model
    #     dynamics_loss(network, batch, model)

    def loss():
        total_loss = 0
        for model in network.dynamic_network.models:
            total_loss += dynamics_loss(network, batch, model)

        diversity_loss = theil_index_loss(network.dynamic_network.models)
        weighted_diversity_loss = config.diversity_loss_weight * diversity_loss
        total_loss += weighted_diversity_loss
        return loss

    variables = [variables
            for variables_list in map(lambda n: n.trainable_weights, network.dynamic_network.models)
            for variables in variables_list]
    var_list_fn = lambda: variables
    optimizer.minimize(loss=loss, var_list=var_list_fn)



def dynamics_loss(network: BaseNetwork, batch, dynamics_model):
    loss = 0
    image_batch, _, targets_time_batch, actions_time_batch, mask_time_batch, dynamic_mask_time_batch = batch

    # Initial step, from the real observation: representation + prediction networks
    representation_batch, _, _ = network.initial_model(np.array(image_batch))

    # Recurrent steps, from action and previous hidden state.
    for actions_batch, targets_batch, mask, dynamic_mask in zip(actions_time_batch, targets_time_batch,
                                                                mask_time_batch, dynamic_mask_time_batch):
        _, _, _, target_next_state_batch = zip(*targets_batch)
        # Compute hidden state representation of next state. This will be
        # used to compute consistency loss.
        target_representation_batch = network.representation_network(np.array(target_next_state_batch))
        target_representation_batch = tf.stop_gradient(target_representation_batch)
        target_representation_batch = tf.boolean_mask(target_representation_batch, mask)

        # Only execute BPTT for elements with an action
        representation_batch = tf.boolean_mask(representation_batch, dynamic_mask)
        # Creating conditioned_representation: concatenate representations with actions batch
        actions_batch = tf.one_hot(actions_batch, network.action_size)

        # Recurrent step from conditioned representation: recurrent + prediction networks
        conditioned_representation_batch = tf.concat((representation_batch, actions_batch), axis=1)

        representation_batch = dynamics_model(conditioned_representation_batch)

        # Only execute BPTT for elements with a policy target
        consistency_loss = tf.math.reduce_mean(tf.math.squared_difference(representation_batch, target_representation_batch))

        # Scale the gradient of the loss by the average number of actions unrolled
        gradient_scale = 1. / len(actions_time_batch)
        loss += scale_gradient(consistency_loss, gradient_scale)

        # Half the gradient of the representation
        representation_batch = scale_gradient(representation_batch, 0.5)

    return loss

    # var_list_fn = lambda: dynamics_model.trainable_weights
    # optimizer.minimize(loss=loss, var_list=var_list_fn)


def update_weights(optimizer: tf.keras.optimizers, network: BaseNetwork, accumulator: Accumulator, batch):
    def loss():
        loss = 0
        image_batch, targets_init_batch, targets_time_batch, actions_time_batch, mask_time_batch, dynamic_mask_time_batch = batch

        # Initial step, from the real observation: representation + prediction networks
        representation_batch, value_batch, policy_batch = network.initial_model(np.array(image_batch))

        # Only update the element with a policy target
        target_value_batch, _, target_policy_batch, _ = zip(*targets_init_batch)
        mask_policy = list(map(lambda l: bool(l), target_policy_batch))
        target_policy_batch = list(filter(lambda l: bool(l), target_policy_batch))
        policy_batch = tf.boolean_mask(policy_batch, mask_policy)

        # Compute the loss of the first pass
        loss += tf.math.reduce_mean(loss_value(target_value_batch, value_batch, network.value_support_size))
        loss += tf.math.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=policy_batch, labels=target_policy_batch))

        # Recurrent steps, from action and previous hidden state.
        for actions_batch, targets_batch, mask, dynamic_mask in zip(actions_time_batch, targets_time_batch,
                                                                    mask_time_batch, dynamic_mask_time_batch):
            target_value_batch, target_reward_batch, target_policy_batch, target_next_state_batch = zip(*targets_batch)
            # Compute hidden state representation of next state. This will be
            # used to compute consistency loss.
            target_representation_batch = network.representation_network(np.array(target_next_state_batch))
            target_representation_batch = tf.stop_gradient(target_representation_batch)

            # Only execute BPTT for elements with an action
            representation_batch = tf.boolean_mask(representation_batch, dynamic_mask)
            target_value_batch = tf.boolean_mask(target_value_batch, mask)
            target_reward_batch = tf.boolean_mask(target_reward_batch, mask)
            target_representation_batch = tf.boolean_mask(target_representation_batch, mask)
            # Creating conditioned_representation: concatenate representations with actions batch
            actions_batch = tf.one_hot(actions_batch, network.action_size)

            # Recurrent step from conditioned representation: recurrent + prediction networks
            conditioned_representation_batch = tf.concat((representation_batch, actions_batch), axis=1)

            representation_batch, reward_batch, value_batch, policy_batch, uncertainty_batch = network.recurrent_model(
                    conditioned_representation_batch)
            accumulator.add(uncertainty_batch)

            # Only execute BPTT for elements with a policy target
            target_policy_batch = [policy for policy, b in zip(target_policy_batch, mask) if b]
            mask_policy = list(map(lambda l: bool(l), target_policy_batch))
            target_policy_batch = tf.convert_to_tensor([policy for policy in target_policy_batch if policy])
            policy_batch = tf.boolean_mask(policy_batch, mask_policy)

            # Compute the partial loss
            l = (tf.math.reduce_mean(loss_value(target_value_batch, value_batch, network.value_support_size)) +
                 MSE(target_reward_batch, tf.squeeze(reward_batch)) +
                 tf.math.reduce_mean(
                     tf.nn.softmax_cross_entropy_with_logits(logits=policy_batch, labels=target_policy_batch)))

            # Scale the gradient of the loss by the average number of actions unrolled
            gradient_scale = 1. / len(actions_time_batch)
            loss += scale_gradient(l, gradient_scale)

            # Half the gradient of the representation
            representation_batch = scale_gradient(representation_batch, 0.5)

        return loss

    optimizer.minimize(loss=loss, var_list=network.cb_get_variables())
    network.training_steps += 1


def loss_value(target_value_batch, value_batch, value_support_size: int):
    batch_size = len(target_value_batch)
    targets = np.zeros((batch_size, value_support_size))
    sqrt_value = np.sqrt(np.abs(target_value_batch)) * np.sign(target_value_batch)
    floor_value = np.floor(sqrt_value).astype(int)
    rest = sqrt_value - floor_value
    targets[range(batch_size), floor_value.astype(int)] = 1 - rest
    targets[range(batch_size), floor_value.astype(int) + 1] = rest

    return tf.nn.softmax_cross_entropy_with_logits(logits=value_batch, labels=targets)

def theil_index_loss(models: List[Model]) -> float:
    weights = [model.get_weights() for model in models]
    total_entropy = 0
    num_layers = len(models[0].get_weights())
    for layer_idx in range(num_layers):
        layer_weights = [weight[layer_idx] for weight in weights]
        total_entropy += layer_entropy(layer_weights)
    # Return a negative value because we want to increase entropy and encourage diveristy
    return -total_entropy / num_layers

def layer_entropy(layer_weights) -> float:
    weight_norms = [tf.norm(layer_weight) for layer_weight in layer_weights]
    mean_norm = sum(weight_norms) / len(weight_norms)
    layer_weight_entropies = [entropy(norm / mean_norm) for norm in weight_norms]
    return sum(layer_weight_entropies) / len(layer_weight_entropies)

def entropy(value: float) -> float:
    return value * math.log(value)
