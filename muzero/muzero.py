import argparse
import tensorflow as tf

from config import MuZeroConfig, default_cartpole_config, consistency_cartpole_config, ensemble_dynamics_cartpole_config, uncertainty_exploration_cartpole_config, full_uncertainty_exploration_cartpole_config, uncertainty_exploration_and_diversity_cartpole_config, default_minigrid_config, uncertainty_exploration_and_diversity_grid_config
from networks.shared_storage import SharedStorage
from self_play.self_play import run_selfplay, run_eval
from training.replay_buffer import ReplayBuffer
from training.training import train_network
from training.ensemble_training import train_ensemble_network


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='''The Muzero configuration to run.''', default="DEFAULT")


def muzero(config: MuZeroConfig):
    """
    MuZero training is split into two independent parts: Network training and
    self-play data generation.
    These two parts only communicate by transferring the latest networks checkpoint
    from the training to the self-play, and the finished games from the self-play
    to the training.
    In contrast to the original MuZero algorithm this version doesn't works with
    multiple threads, therefore the training and self-play is done alternately.
    """
    # Disable logging for interactive training
    if 'disable_interactive_logging' in dir(tf.keras.utils):
        tf.keras.utils.disable_interactive_logging()

    storage = SharedStorage(config.new_network(), config.uniform_network(), config.new_optimizer())
    replay_buffer = ReplayBuffer(config)

    for loop in range(config.nb_training_loop):
        print("Training loop", loop)
        score_train = run_selfplay(config, storage, replay_buffer, config.nb_episodes)
        print("Train score:", score_train)
        avg_uncertainty = train_ensemble_network(config, storage, replay_buffer, config.nb_epochs)
        print("Uncertainty score after training:", avg_uncertainty)

        print(f"MuZero played {config.nb_episodes * (loop + 1)} "
              f"episodes and trained for {config.nb_epochs * (loop + 1)} epochs.\n")

    return storage.latest_network()


if __name__ == '__main__':
    args = parser.parse_args()
    config_mapping = {
        "DEFAULT": default_cartpole_config(),
        "CONSISTENCY": consistency_cartpole_config(),
        "ENSEMBLE_CONSISTENCY": ensemble_dynamics_cartpole_config(),
        "ENSEMBLE_CONSISTENCY_WITH_EXPLORATION": uncertainty_exploration_cartpole_config(),
        "ENSEMBLE_CONSISTENCY_PURE_EXPLORATION": full_uncertainty_exploration_cartpole_config(),
        "ENSEMBLE_CONSISTENCY_WITH_EXPLORATION_AND_DIVERSITY": uncertainty_exploration_and_diversity_cartpole_config(),
        "MINIGRID_DEFAULT": default_minigrid_config(),
        "MINIGRID_ENSEMBLE_CONSISTENCY_WITH_EXPLORATION_AND_DIVERSITY": uncertainty_exploration_and_diversity_grid_config(),
    }
    config = config_mapping.get(args.config, None)
    if config:
        muzero(config)
    else:
        print("Invalid muzero config provided: {0}".format(args.config))
