"""
Training script for 13×13 FX+WDLP AlphaZero Gomoku (Enhanced version).

This is the FX+WDLP enhanced version for comparison against the baseline.
Same network architecture, same compute — only addition is FX features
and WDLP value head.

Designed to run SIMULTANEOUSLY with training_gomoku_13x13_baseline.py on a
single A800 80GB GPU (12 actors each, ~40GB per run).

Usage:
    python training_gomoku_13x13.py
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import multiprocessing as mp
import sys
from absl import flags

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR

FLAGS = flags.FLAGS

# Board and Network (same arch as baseline for fair comparison)
flags.DEFINE_integer('board_size', 13, 'Board size for freestyle Gomoku.')
flags.DEFINE_integer('num_stack', 4, 'Stack N previous states.')
flags.DEFINE_integer('num_res_blocks', 8, 'Number of residual blocks.')
flags.DEFINE_integer('num_filters', 96, 'Number of filters for conv2d layers.')
flags.DEFINE_integer('num_fc_units', 96, 'Number of hidden units in linear layer.')

# FX + WDLP ENABLED — this is the innovation
flags.DEFINE_bool('use_fx', True, 'FX features ENABLED.')
flags.DEFINE_bool('use_wdlp', True, 'WDLP value head ENABLED.')
flags.DEFINE_float('wdl_loss_weight', 0.01, 'Weight for WDL loss (alpha).')
flags.DEFINE_float('plies_loss_weight', 0.0005, 'Weight for plies loss (beta, tuned for 13×13 game length).')

# Training Configuration (identical to baseline)
flags.DEFINE_integer('min_games', 1000, 'Minimum self-play games before learning starts.')
flags.DEFINE_integer('games_per_ckpt', 500, 'Games per checkpoint.')
flags.DEFINE_integer('replay_capacity', 250000, 'Replay buffer capacity.')
flags.DEFINE_integer('batch_size', 512, 'Training batch size.')
flags.DEFINE_float('init_lr', 0.001, 'Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, 'Learning rate decay rate.')
flags.DEFINE_multi_integer('lr_milestones', [30000, 45000], 'LR decay milestones.')
flags.DEFINE_float('sgd_momentum', 0.9, 'SGD momentum.')
flags.DEFINE_float('l2_regularization', 1e-4, 'L2 regularization parameter.')
flags.DEFINE_integer('max_training_steps', 25000, 'Maximum training steps.')
flags.DEFINE_bool('argument_data', True, 'Apply random rotation and mirroring.')
flags.DEFINE_bool('compress_data', False, 'Compress state in replay buffer.')

# MCTS Configuration (identical to baseline)
flags.DEFINE_integer('num_actors', 12, 'Number of self-play actors (12 for parallel run).')
flags.DEFINE_integer('num_simulations', 100, 'MCTS simulations per move.')
flags.DEFINE_integer('num_parallel', 8, 'Parallel leaf collection during MCTS.')
flags.DEFINE_float('c_puct_base', 19652, 'MCTS exploration constant base.')
flags.DEFINE_float('c_puct_init', 1.25, 'MCTS exploration constant init.')
flags.DEFINE_integer('warm_up_steps', 12, 'Steps with temperature=1 at game start.')

# Resign settings (Not used for Gomoku)
flags.DEFINE_float('init_resign_threshold', -1, 'Not applicable for Gomoku.')
flags.DEFINE_integer('check_resign_after_steps', 0, 'Not applicable.')
flags.DEFINE_float('target_fp_rate', 0, 'Not applicable.')
flags.DEFINE_float('disable_resign_ratio', 0, 'Not applicable.')
flags.DEFINE_integer('reset_fp_interval', 0, 'Not applicable.')
flags.DEFINE_integer('no_resign_games', 0, 'Not applicable.')

# Logging and Checkpointing
flags.DEFINE_float('default_rating', 0, 'Default elo rating.')
flags.DEFINE_integer('ckpt_interval', 500, 'Checkpoint frequency (training steps).')
flags.DEFINE_integer('log_interval', 100, 'Logging frequency (training steps).')
flags.DEFINE_string('ckpt_dir', './checkpoints/gomoku/13x13_fxwdlp', 'Checkpoint directory.')
flags.DEFINE_string('logs_dir', './logs/gomoku/13x13_fxwdlp', 'Logs directory.')
flags.DEFINE_string('eval_games_dir', '', 'Evaluation games directory.')
flags.DEFINE_string('save_sgf_dir', './games/selfplay_games/gomoku/13x13_fxwdlp', 'SGF save directory.')
flags.DEFINE_integer('save_sgf_interval', 500, 'Save self-play games interval.')
flags.DEFINE_integer('save_replay_interval', 0, 'Save replay buffer interval.')
flags.DEFINE_string('load_ckpt', '', 'Resume from checkpoint path.')
flags.DEFINE_string('load_replay', '', 'Resume from replay buffer path.')
flags.DEFINE_string('log_level', 'INFO', 'Logging level.')
flags.DEFINE_integer('seed', 1, 'Random seed.')

# Validators
flags.register_validator('num_simulations', lambda x: x > 1)
flags.register_validator('init_resign_threshold', lambda x: x <= -1)
flags.register_validator('log_level', lambda x: x in ['INFO', 'DEBUG'])
flags.register_multi_flags_validator(
    ['num_parallel', 'c_puct_base'],
    lambda flags: flags['c_puct_base'] >= 19652 * (flags['num_parallel'] / 800),
    '',
)

FLAGS(sys.argv)

from alpha_zero.envs.gomoku import GomokuEnv
from alpha_zero.core.pipeline import (
    run_learner_loop,
    run_evaluator_loop,
    run_selfplay_actor_loop,
    set_seed,
    maybe_create_dir,
)
from alpha_zero.core.network import AlphaZeroNet
from alpha_zero.core.replay import UniformReplay
from alpha_zero.utils.util import extract_args_from_flags_dict, create_logger


def main():
    set_seed(FLAGS.seed)

    maybe_create_dir(FLAGS.ckpt_dir)
    maybe_create_dir(FLAGS.logs_dir)
    maybe_create_dir(FLAGS.save_sgf_dir)

    logger = create_logger(FLAGS.log_level)
    logger.info('=' * 60)
    logger.info('  13×13 FX+WDLP AlphaZero Gomoku (Enhanced)')
    logger.info('=' * 60)
    logger.info(extract_args_from_flags_dict(FLAGS.flag_values_dict()))

    # Device configuration
    actor_devices = [torch.device('cpu')] * FLAGS.num_actors
    learner_device = eval_device = torch.device('cpu')

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            learner_device = torch.device(f'cuda:{num_gpus-1}')
            eval_device = torch.device(f'cuda:{num_gpus-2}')
        else:
            learner_device = eval_device = torch.device('cuda:0')

        actor_devices = [torch.device(f'cuda:{i % num_gpus}') for i in range(FLAGS.num_actors)]

        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f'GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)')

    def env_builder():
        return GomokuEnv(board_size=FLAGS.board_size, num_stack=FLAGS.num_stack)

    eval_env = env_builder()
    input_shape = eval_env.observation_space.shape

    # FX features: double input channels (9 → 18)
    if FLAGS.use_fx:
        c, h, w = input_shape
        input_shape = (c * 2, h, w)
        logger.info(f'FX features enabled: input shape {input_shape}')

    num_actions = eval_env.action_space.n
    logger.info(f'Board: {FLAGS.board_size}×{FLAGS.board_size}, Actions: {num_actions}')

    def network_builder():
        return AlphaZeroNet(
            input_shape,
            num_actions,
            FLAGS.num_res_blocks,
            FLAGS.num_filters,
            FLAGS.num_fc_units,
            gomoku=True,
            use_wdlp=FLAGS.use_wdlp,  # ENHANCED: WDLP value head
        )

    network = network_builder()
    total_params = sum(p.numel() for p in network.parameters())
    logger.info(f'Network: {FLAGS.num_res_blocks} res blocks, {FLAGS.num_filters} filters')
    logger.info(f'Parameters: {total_params:,} (with WDLP head)')

    optimizer = torch.optim.SGD(
        network.parameters(),
        lr=FLAGS.init_lr,
        momentum=FLAGS.sgd_momentum,
        weight_decay=FLAGS.l2_regularization,
    )
    lr_scheduler = MultiStepLR(optimizer, milestones=FLAGS.lr_milestones, gamma=FLAGS.lr_decay)

    stop_event = mp.Event()
    ckpt_event = mp.Event()
    data_queue = mp.Queue(maxsize=FLAGS.num_actors)

    with mp.Manager() as manager:
        var_ckpt = manager.Value('s', b'')
        var_resign_threshold = manager.Value('d', FLAGS.init_resign_threshold)

        replay = UniformReplay(
            capacity=FLAGS.replay_capacity,
            random_state=np.random.RandomState(),
            compress_data=FLAGS.compress_data,
            use_wdlp=FLAGS.use_wdlp,  # ENHANCED: WDLP support
        )

        # Start evaluator
        evaluator = mp.Process(
            target=run_evaluator_loop,
            args=(
                FLAGS.seed,
                network_builder(),
                eval_device,
                eval_env,
                FLAGS.eval_games_dir,
                FLAGS.num_simulations,
                FLAGS.num_parallel,
                FLAGS.c_puct_base,
                FLAGS.c_puct_init,
                FLAGS.default_rating,
                FLAGS.logs_dir,
                FLAGS.save_sgf_dir,
                FLAGS.load_ckpt,
                FLAGS.log_level,
                var_ckpt,
                stop_event,
                FLAGS.use_fx,        # use_fx = True
                FLAGS.board_size,    # board_size = 13
            ),
        )
        evaluator.start()

        # Start self-play actors
        actors = []
        for i in range(FLAGS.num_actors):
            actor = mp.Process(
                target=run_selfplay_actor_loop,
                args=(
                    FLAGS.seed,
                    i,
                    network_builder(),
                    actor_devices[i],
                    data_queue,
                    env_builder(),
                    FLAGS.num_simulations,
                    FLAGS.num_parallel,
                    FLAGS.c_puct_base,
                    FLAGS.c_puct_init,
                    FLAGS.warm_up_steps,
                    FLAGS.check_resign_after_steps,
                    FLAGS.disable_resign_ratio,
                    FLAGS.save_sgf_dir,
                    FLAGS.save_sgf_interval,
                    FLAGS.logs_dir,
                    FLAGS.load_ckpt,
                    FLAGS.log_level,
                    var_ckpt,
                    var_resign_threshold,
                    ckpt_event,
                    stop_event,
                    FLAGS.use_fx,        # use_fx = True
                    FLAGS.board_size,    # board_size = 13
                ),
            )
            actor.start()
            actors.append(actor)

        # Run learner loop
        run_learner_loop(
            seed=FLAGS.seed,
            network=network,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=learner_device,
            replay=replay,
            logger=logger,
            argument_data=FLAGS.argument_data,
            batch_size=FLAGS.batch_size,
            init_resign_threshold=FLAGS.init_resign_threshold,
            disable_resign_ratio=FLAGS.disable_resign_ratio,
            target_fp_rate=FLAGS.target_fp_rate,
            reset_fp_interval=FLAGS.reset_fp_interval,
            no_resign_games=FLAGS.no_resign_games,
            min_games=FLAGS.min_games,
            games_per_ckpt=FLAGS.games_per_ckpt,
            num_actors=FLAGS.num_actors,
            ckpt_interval=FLAGS.ckpt_interval,
            log_interval=FLAGS.log_interval,
            save_replay_interval=FLAGS.save_replay_interval,
            max_training_steps=FLAGS.max_training_steps,
            ckpt_dir=FLAGS.ckpt_dir,
            logs_dir=FLAGS.logs_dir,
            load_ckpt=FLAGS.load_ckpt,
            load_replay=FLAGS.load_replay,
            data_queue=data_queue,
            var_ckpt=var_ckpt,
            var_resign_threshold=var_resign_threshold,
            ckpt_event=ckpt_event,
            stop_event=stop_event,
            use_fx=FLAGS.use_fx,
            use_wdlp=FLAGS.use_wdlp,
            wdl_loss_weight=FLAGS.wdl_loss_weight,
            plies_loss_weight=FLAGS.plies_loss_weight,
            board_size=FLAGS.board_size,
        )

        for actor in actors:
            actor.join()
            actor.close()
        evaluator.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
