import os
from typing import Any, Text, Callable, Mapping, Iterable, Tuple
import time
from pathlib import Path
from collections import OrderedDict, deque
import queue
import multiprocessing as mp
import threading
import pickle
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

import numpy as np
from copy import copy, deepcopy

from alpha_zero.core.mcts_v2 import Node, parallel_uct_search, uct_search
from alpha_zero.envs.base import BoardGameEnv
from alpha_zero.core.eval_dataset import build_eval_dataset
from alpha_zero.core.rating import EloRating
from alpha_zero.core.replay import UniformReplay, Transition, value_to_wdl
from alpha_zero.utils.csv_writer import CsvWriter
from alpha_zero.utils.transformation import apply_random_transformation
from alpha_zero.utils.util import create_logger, get_time_stamp


# =================================================================
# Helper Functions
# =================================================================

def disable_auto_grad(network: torch.nn.Module) -> None:
    for p in network.parameters():
        p.requires_grad = False


def set_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def maybe_create_dir(dir) -> None:
    if dir is not None and dir != '' and not os.path.exists(dir):
        Path(dir).mkdir(parents=True, exist_ok=False)


def save_to_file(obj: Any, file_name: str) -> None:
    pickle.dump(obj, open(file_name, 'wb'))


def load_from_file(file_name: str) -> Any:
    return pickle.load(open(file_name, 'rb'))


def round_it(v, places=4) -> float:
    return round(v, places)


def _encode_bytes(in_str) -> Any:
    return str(in_str).encode('utf-8')


def _decode_bytes(b) -> str:
    return b.decode('utf-8')


class SimpleTimer:
    """
    Simple timer class that can be used as a context manager.
    
    This is a robust implementation that works with the existing code.
    """
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self._elapsed = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self._elapsed = self.end_time - self.start_time
        return False
    
    @property
    def elapsed(self):
        """Return elapsed time in seconds."""
        if self._elapsed > 0:
            return self._elapsed
        elif self.start_time is not None:
            return time.time() - self.start_time
        else:
            return 0.0


def _compute_threat_map_vectorized(stones: np.ndarray) -> np.ndarray:
    """
    Compute threat map using vectorized numpy operations (3×3 dilation).
    
    Works efficiently for any board size (7×7, 15×15, etc.).
    Replaces the slow nested Python loops in the original implementation.
    
    Args:
        stones: Binary stone positions [batch, 1, h, w]
        
    Returns:
        threat_map: Dilated threat map [batch, 1, h, w]
    """
    batch, _, h, w = stones.shape
    # Pad with zeros, then sum 3×3 shifted windows
    padded = np.pad(stones, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
    threat_map = np.zeros_like(stones)
    for di in range(3):
        for dj in range(3):
            threat_map += padded[:, :, di:di+h, dj:dj+w]
    return (threat_map > 0).astype(stones.dtype)


# Pre-compute checkerboard patterns for common board sizes (cached)
_CHECKERBOARD_CACHE = {}

def _get_checkerboard(h: int, w: int, dtype) -> np.ndarray:
    """Get cached checkerboard pattern [1, 1, h, w]."""
    key = (h, w, dtype)
    if key not in _CHECKERBOARD_CACHE:
        rows = np.arange(h).reshape(h, 1)
        cols = np.arange(w).reshape(1, w)
        pattern = ((rows + cols) % 2).astype(dtype)
        _CHECKERBOARD_CACHE[key] = pattern.reshape(1, 1, h, w)
    return _CHECKERBOARD_CACHE[key]


def create_enhanced_input(state: np.ndarray, board_size: int = 7) -> np.ndarray:
    """
    Create enhanced FX input representation for Gomoku.
    
    INNOVATION #1: Enhanced input features based on Czech et al. (2024).
    Adapted from chess FX representation to Gomoku.
    
    Optimized with vectorized numpy operations for efficient computation
    on both 7×7 and 15×15 boards.
    
    Args:
        state: Original state [batch, C, h, w] or [C, h, w] where C varies (8 or 9 channels)
        board_size: Board size (7 for 7×7, 15 for 15×15 Gomoku)
        
    Returns:
        enhanced: Enhanced state [batch, C*2, h, w] or [C*2, h, w]
        
    New Features (C additional channels matching original):
        Checkerboard pattern, piece masks, stone counts, threat map, game phase, etc.
    """
    # Handle both batched and unbatched inputs
    if state.ndim == 3:
        state = state[np.newaxis, ...]
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch, channels, h, w = state.shape
    assert h == board_size and w == board_size
    
    # Current position (most recent timestep)
    p1_current = state[:, 0:1, :, :]
    p2_current = state[:, 1:2, :, :]
    
    enhanced_features = [state]  # Keep ALL original channels
    
    # Calculate how many additional channels we need to match original count
    num_additional = channels
    
    # +0: Checkerboard pattern (vectorized, cached)
    checkerboard = np.broadcast_to(
        _get_checkerboard(h, w, state.dtype), (batch, 1, h, w)
    ).copy()
    enhanced_features.append(checkerboard)
    
    # +1, +2: Piece masks
    enhanced_features.append(p1_current.copy())
    enhanced_features.append(p2_current.copy())
    
    # +3, +4, +5: Material counts (normalized by board area)
    max_stones = h * w
    p1_count = np.sum(p1_current, axis=(2, 3), keepdims=True) / max_stones
    p2_count = np.sum(p2_current, axis=(2, 3), keepdims=True) / max_stones
    count_diff = p1_count - p2_count
    
    enhanced_features.append(np.broadcast_to(count_diff, (batch, 1, h, w)).copy())
    enhanced_features.append(np.broadcast_to(p1_count, (batch, 1, h, w)).copy())
    enhanced_features.append(np.broadcast_to(p2_count, (batch, 1, h, w)).copy())
    
    # +6: Threat map (vectorized 3×3 dilation — replaces slow nested loops)
    threat_map = _compute_threat_map_vectorized(p1_current)
    enhanced_features.append(threat_map)
    
    # +7: Game phase (normalized total stone count)
    total_stones = (p1_count + p2_count) * max_stones
    game_phase = total_stones / max_stones
    enhanced_features.append(np.broadcast_to(game_phase, (batch, 1, h, w)).copy())
    
    # +8+: Empty squares indicator and padding channels
    current_added = 8
    while current_added < num_additional:
        if current_added == 8:
            empty_map = 1.0 - np.clip(p1_current + p2_current, 0.0, 1.0)
            enhanced_features.append(empty_map)
        else:
            enhanced_features.append(np.zeros((batch, 1, h, w), dtype=state.dtype))
        current_added += 1
    
    # Concatenate all features
    enhanced = np.concatenate(enhanced_features, axis=1)
    
    if squeeze_output:
        enhanced = enhanced[0]
    
    return enhanced


def create_mcts_player(
    network: torch.nn.Module,
    device: torch.device,
    num_simulations: int,
    num_parallel: int,
    root_noise: bool = False,
    deterministic: bool = False,
    use_fx: bool = False,
    board_size: int = 7,
) -> Callable[[BoardGameEnv, Node, float, float, bool], Tuple[int, np.ndarray, float, float, Node]]:
    """Create MCTS player with neural network evaluation."""
    
    @torch.no_grad()
    def eval_position(state: np.ndarray, batched: bool = False) -> Tuple[Iterable[np.ndarray], Iterable[float]]:
        """Evaluate position using neural network."""
        if not batched:
            state = state[None, ...]

        # INNOVATION: Apply FX preprocessing if enabled
        if use_fx:
            state = create_enhanced_input(state, board_size)

        state = torch.from_numpy(state).to(dtype=torch.float32, device=device, non_blocking=True)
        
        # Network forward pass
        outputs = network(state)
        if len(outputs) == 4:  # WDLP mode
            pi_logits, v, wdl, plies = outputs
        else:  # Standard mode
            pi_logits, v = outputs

        pi_logits = torch.detach(pi_logits)
        v = torch.detach(v)

        pi = torch.softmax(pi_logits, dim=-1).cpu().numpy()
        v = v.cpu().numpy()

        B, *_ = state.shape
        v = np.squeeze(v, axis=1).tolist()
        pi = [pi[i] for i in range(B)]

        if not batched:
            pi = pi[0]
            v = v[0]

        return pi, v

    def act(
        env: BoardGameEnv,
        root_node: Node,
        c_puct_base: float,
        c_puct_init: float,
        warm_up: bool = False,
    ) -> Tuple[int, np.ndarray, float, float, Node]:
        """Perform MCTS and select action."""
        if num_parallel > 1:
            return parallel_uct_search(
                env=env,
                eval_func=eval_position,
                root_node=root_node,
                c_puct_base=c_puct_base,
                c_puct_init=c_puct_init,
                num_simulations=num_simulations,
                num_parallel=num_parallel,
                root_noise=root_noise,
                warm_up=warm_up,
                deterministic=deterministic,
            )
        else:
            return uct_search(
                env=env,
                eval_func=eval_position,
                root_node=root_node,
                c_puct_base=c_puct_base,
                c_puct_init=c_puct_init,
                num_simulations=num_simulations,
                root_noise=root_noise,
                warm_up=warm_up,
                deterministic=deterministic,
            )

    return act


# =================================================================
# Self-Play
# =================================================================

def run_selfplay_actor_loop(
    seed: int,
    rank: int,
    network: torch.nn.Module,
    device: torch.device,
    data_queue: mp.Queue,
    env: BoardGameEnv,
    num_simulations: int,
    num_parallel: int,
    c_puct_base: float,
    c_puct_init: float,
    warm_up_steps: int,
    check_resign_after_steps: int,
    disable_resign_ratio: float,
    save_sgf_dir: str,
    save_sgf_interval: int,
    logs_dir: str,
    load_ckpt: str,
    log_level: str,
    var_ckpt: mp.Value,
    var_resign_threshold: mp.Value,
    ckpt_event: mp.Event,
    stop_event: mp.Event,
    use_fx: bool = False,
    board_size: int = 7,
) -> None:
    """Self-play actor: generates training data through self-play games."""
    assert num_simulations > 1

    set_seed(int(seed + rank))
    logger = create_logger(log_level)
    writer = CsvWriter(os.path.join(logs_dir, f'actor{rank}.csv'))
    timer = SimpleTimer()

    played_games = training_steps = 0
    last_ckpt = None
    should_save_sgf = save_sgf_dir and os.path.isdir(save_sgf_dir)

    disable_auto_grad(network)
    network = network.to(device=device)

    if load_ckpt and os.path.exists(load_ckpt):
        loaded_state = torch.load(load_ckpt, map_location=device)
        network.load_state_dict(loaded_state['network'])
        training_steps = loaded_state['training_steps']
        logger.debug(f'Actor{rank} loaded checkpoint "{load_ckpt}"')

    network.eval()

    resign_threshold = var_resign_threshold.value if env.has_resign_move else -1
    mcts_player = create_mcts_player(
        network=network,
        device=device,
        num_simulations=num_simulations,
        num_parallel=num_parallel,
        root_noise=True,
        deterministic=False,
        use_fx=use_fx,
        board_size=board_size,
    )

    while not stop_event.is_set():
        if ckpt_event.is_set():
            continue

        # Load new checkpoint if available
        new_ckpt = _decode_bytes(var_ckpt.value)
        if new_ckpt and new_ckpt != last_ckpt and os.path.exists(new_ckpt):
            loaded_state = torch.load(new_ckpt, map_location=device)
            network.load_state_dict(loaded_state['network'])
            training_steps = loaded_state['training_steps']
            network.eval()
            last_ckpt = new_ckpt
            logger.debug(f'Actor{rank} switched to checkpoint "{new_ckpt}"')

        if env.has_resign_move:
            resign_threshold = var_resign_threshold.value

        resign_disabled = True
        if env.has_resign_move and resign_threshold > -1.0 and np.random.rand() > disable_resign_ratio:
            resign_disabled = False

        with timer:
            game_seq, stats = play_and_record_one_game(
                env=env,
                mcts_player=mcts_player,
                resign_disabled=resign_disabled,
                c_puct_base=c_puct_base,
                c_puct_init=c_puct_init,
                warm_up_steps=warm_up_steps,
                check_resign_after_steps=check_resign_after_steps,
                resign_threshold=resign_threshold,
                board_size=board_size,
            )

        played_games += 1

        # Send game data to learner
        try:
            data_queue.put(game_seq, block=True, timeout=10)
        except queue.Full:
            logger.warning(f'Actor{rank} timeout sending samples to learner')

        stats['actor_id'] = rank
        stats['datetime'] = get_time_stamp()
        stats['training_steps'] = training_steps
        stats['played_games'] = played_games
        stats['game_time'] = round_it(timer.elapsed, 3)
        writer.write(OrderedDict((n, v) for n, v in stats.items()))

        # Save SGF periodically
        if should_save_sgf and played_games % save_sgf_interval == 0:
            sgf_file = os.path.join(save_sgf_dir, f'actor{rank}_game{played_games}.sgf')
            with open(sgf_file, 'w') as f:
                f.write(env.to_sgf())

    writer.close()


def play_and_record_one_game(
    env: BoardGameEnv,
    mcts_player: Callable,
    resign_disabled: bool,
    c_puct_base: float,
    c_puct_init: float,
    warm_up_steps: int,
    check_resign_after_steps: int,
    resign_threshold: float,
    board_size: int = 7,
) -> Tuple[list, Mapping[Text, Any]]:
    """
    Play one self-play game and record transitions.
    
    PREVIOUS INNOVATION #2: Proper value assignment from game outcomes.
    """
    obs = env.reset()
    done = False
    game_seq = []
    root_node = None
    num_passes = 0
    num_resigns = 0
    move_count = 0

    while not done:
        warm_up = env.steps < warm_up_steps
        move, pi, q_value, subtree_value, root_node = mcts_player(
            env=env,
            root_node=root_node,
            c_puct_base=c_puct_base,
            c_puct_init=c_puct_init,
            warm_up=warm_up,
        )

        # Check for resign
        should_resign = False
        if (
            not resign_disabled
            and env.has_resign_move
            and env.steps >= check_resign_after_steps
            and subtree_value <= resign_threshold
        ):
            should_resign = True
            move = env.resign_move
            num_resigns += 1

        # PREVIOUS INNOVATION: Store state, policy, and PLAYER (not value yet)
        # Value will be backfilled after game ends
        game_seq.append((obs, pi, env.to_play, move_count))
        obs, _, done, _ = env.step(move)
        move_count += 1

        if env.has_pass_move and move == env.pass_move:
            num_passes += 1

        # Reuse subtree for next step
        if root_node is not None and move in root_node.children:
            root_node = root_node.children[move]
            root_node.parent = None
        else:
            root_node = None

    # PREVIOUS INNOVATION #2: Backfill values based on game outcome
    # This is the KEY FIX - assign correct values from winner's perspective
    winner = env.winner
    final_game_seq = []
    
    for obs, pi, player, move_idx in game_seq:
        # Assign value from current player's perspective
        if winner is None:
            value = 0.0  # Draw
        elif winner == player:
            value = 1.0  # Win for this player
        else:
            value = -1.0  # Loss for this player
        
        # Calculate remaining plies (actual moves until game end)
        total_game_length = len(game_seq)
        remaining_plies = max(0, total_game_length - move_idx)
        
        # Create proper transition with WDL target
        wdl_target = value_to_wdl(value)
        
        final_game_seq.append(Transition(
            state=obs,
            pi_prob=pi,
            value=value,
            wdl=wdl_target,
            plies=float(remaining_plies),
        ))

    stats = {
        'game_length': env.steps,
        'game_result': env.get_result_string(),
    }

    if env.has_pass_move:
        stats['num_passes'] = num_passes
    if env.has_resign_move:
        stats['num_resigns'] = num_resigns

    return final_game_seq, stats


# =================================================================
# Learner
# =================================================================

def run_learner_loop(
    seed: int,
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    replay: UniformReplay,
    logger: Any,
    argument_data: bool,
    batch_size: int,
    init_resign_threshold: float,
    disable_resign_ratio: float,
    target_fp_rate: float,
    reset_fp_interval: int,
    no_resign_games: int,
    min_games: int,
    games_per_ckpt: int,
    num_actors: int,
    ckpt_interval: int,
    log_interval: int,
    save_replay_interval: int,
    max_training_steps: int,
    ckpt_dir: str,
    logs_dir: str,
    load_ckpt: str,
    load_replay: str,
    data_queue: mp.Queue,
    var_ckpt: mp.Value,
    var_resign_threshold: mp.Value,
    ckpt_event: mp.Event,
    stop_event: mp.Event,
    use_fx: bool = False,
    use_wdlp: bool = False,
    wdl_loss_weight: float = 0.01,
    plies_loss_weight: float = 0.002,
    board_size: int = 7,
) -> None:
    """
    Main training loop: receives self-play data and trains the network.
    
    INNOVATION #2 (NEW): WDLP loss function for enhanced value prediction.
    """
    set_seed(seed)
    
    network = network.to(device=device)
    network.train()

    writer = CsvWriter(os.path.join(logs_dir, 'training.csv'))
    timer = SimpleTimer()

    training_steps = 0
    num_games_received = 0
    num_transitions_received = 0
    
    # Load checkpoint if resuming
    if load_ckpt and os.path.exists(load_ckpt):
        loaded_state = torch.load(load_ckpt, map_location=device)
        network.load_state_dict(loaded_state['network'])
        optimizer.load_state_dict(loaded_state['optimizer'])
        lr_scheduler.load_state_dict(loaded_state['lr_scheduler'])
        training_steps = loaded_state['training_steps']
        num_games_received = loaded_state['num_games_received']
        logger.info(f'Loaded checkpoint at step {training_steps}')

    # Load replay buffer if resuming
    if load_replay and os.path.exists(load_replay):
        replay_state = load_from_file(load_replay)
        replay.set_state(replay_state)
        logger.info(f'Loaded replay buffer with {replay.size} transitions')

    # Resign tracking
    resign_stats = {
        'fp_count': 0,
        'total_resigns': 0,
        'resign_threshold': init_resign_threshold,
    }
    var_resign_threshold.value = resign_stats['resign_threshold']

    # Start data receiver thread
    def data_receiver():
        nonlocal num_games_received, num_transitions_received
        while not stop_event.is_set():
            try:
                game_seq = data_queue.get(block=True, timeout=1)
                replay.add_game(game_seq)
                num_games_received += 1
                num_transitions_received += len(game_seq)
            except queue.Empty:
                continue

    receiver_thread = threading.Thread(target=data_receiver, daemon=True)
    receiver_thread.start()

    logger.info('Waiting for self-play games to fill replay buffer...')
    while replay.size < min_games * 10:  # Wait for minimum data
        time.sleep(5)
        if stop_event.is_set():
            return

    logger.info(f'Starting training with {replay.size} transitions')
    logger.info(f'FX features: {use_fx}, WDLP value head: {use_wdlp}')

    # Main training loop
    while training_steps < max_training_steps:
        if stop_event.is_set():
            break

        # Wait for enough new games
        if num_games_received < min_games + (training_steps // ckpt_interval) * games_per_ckpt:
            time.sleep(1)
            continue

        # Sample batch and train
        with timer:
            batch = replay.sample(batch_size)
            if batch is None:
                continue
            
            # Apply data augmentation
            if argument_data:
                batch = apply_random_transformation(batch)

            # INNOVATION: Apply FX preprocessing if enabled
            states = batch.state
            if use_fx:
                states = create_enhanced_input(states, board_size)  # Batched — no per-sample loop

            states = torch.from_numpy(states).to(device=device, dtype=torch.float32)
            target_pi = torch.from_numpy(batch.pi_prob).to(device=device, dtype=torch.float32)
            target_v = torch.from_numpy(batch.value).to(device=device, dtype=torch.float32)

            # Forward pass
            outputs = network(states)
            
            if use_wdlp:
                # INNOVATION: WDLP mode
                pi_logits, v, wdl, plies = outputs
                v = v.squeeze(-1)
                
                # WDL and plies targets
                target_wdl = torch.from_numpy(batch.wdl).to(device=device, dtype=torch.float32)
                target_plies = torch.from_numpy(batch.plies).to(device=device, dtype=torch.float32).unsqueeze(1)
                
                # Compute losses
                policy_loss = F.cross_entropy(pi_logits, target_pi)
                wdl_loss = F.cross_entropy(wdl, target_wdl)
                plies_loss = F.mse_loss(plies, target_plies)
                
                # Total loss with weighted components (Czech et al., 2024)
                loss = policy_loss + wdl_loss_weight * wdl_loss + plies_loss_weight * plies_loss
                
                # For logging
                value_loss = wdl_loss  # Report WDL loss as value loss
            else:
                # Standard mode
                pi_logits, v = outputs
                v = v.squeeze(-1)
                
                # Compute losses
                policy_loss = F.cross_entropy(pi_logits, target_pi)
                value_loss = F.mse_loss(v, target_v)
                
                # Total loss
                loss = policy_loss + value_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            optimizer.step()
            lr_scheduler.step()

        training_steps += 1

        # Logging
        if training_steps % log_interval == 0:
            stats = {
                'datetime': get_time_stamp(),
                'training_steps': training_steps,
                'num_games_received': num_games_received,
                'num_transitions_received': num_transitions_received,
                'replay_size': replay.size,
                'policy_loss': round_it(policy_loss.item(), 4),
                'value_loss': round_it(value_loss.item(), 4),
                'total_loss': round_it(loss.item(), 4),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'batch_time': round_it(timer.elapsed, 4),
            }
            
            if use_wdlp:
                stats['wdl_loss'] = round_it(wdl_loss.item(), 4)
                stats['plies_loss'] = round_it(plies_loss.item(), 4)
            
            writer.write(OrderedDict((n, v) for n, v in stats.items()))
            
            if use_wdlp:
                logger.info(
                    f'[Step {training_steps}] Loss: {loss.item():.4f} | '
                    f'Policy: {policy_loss.item():.4f} | WDL: {wdl_loss.item():.4f} | '
                    f'Plies: {plies_loss.item():.4f} | Games: {num_games_received} | Replay: {replay.size}'
                )
            else:
                logger.info(
                    f'[Step {training_steps}] Loss: {loss.item():.4f} | '
                    f'Policy: {policy_loss.item():.4f} | Value: {value_loss.item():.4f} | '
                    f'Games: {num_games_received} | Replay: {replay.size}'
                )

        # Save checkpoint
        if training_steps % ckpt_interval == 0:
            ckpt_event.set()  # Pause actors
            
            ckpt_file = os.path.join(ckpt_dir, f'training_steps_{training_steps}.ckpt')
            torch.save({
                'network': network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'training_steps': training_steps,
                'num_games_received': num_games_received,
            }, ckpt_file)
            
            var_ckpt.value = _encode_bytes(ckpt_file)
            logger.info(f'Saved checkpoint: {ckpt_file}')
            
            ckpt_event.clear()  # Resume actors

        # Save replay buffer periodically
        if save_replay_interval > 0 and num_games_received % save_replay_interval == 0:
            replay_file = os.path.join(ckpt_dir, 'replay_state.ckpt')
            save_to_file(replay.get_state(), replay_file)
            logger.info(f'Saved replay buffer: {replay_file}')

    # Training complete
    logger.info(f'Training completed at step {training_steps}')
    stop_event.set()
    writer.close()


# =================================================================
# Evaluator
# =================================================================

def run_evaluator_loop(
    seed: int,
    network: torch.nn.Module,
    device: torch.device,
    env: BoardGameEnv,
    eval_games_dir: str,
    num_simulations: int,
    num_parallel: int,
    c_puct_base: float,
    c_puct_init: float,
    default_rating: float,
    logs_dir: str,
    save_sgf_dir: str,
    load_ckpt: str,
    log_level: str,
    var_ckpt: mp.Value,
    stop_event: mp.Event,
    use_fx: bool = False,
    board_size: int = 7,
) -> None:
    """Evaluate latest network against previous checkpoint."""
    assert num_simulations > 1

    set_seed(int(seed))
    logger = create_logger(log_level)

    network = network.to(device=device)
    disable_auto_grad(network)

    writer = CsvWriter(os.path.join(logs_dir, 'evaluation.csv'), buffer_size=1)

    last_ckpt = None
    last_ckpt_step = 0

    if load_ckpt and os.path.exists(load_ckpt):
        loaded_state = torch.load(load_ckpt, map_location=device)
        network.load_state_dict(loaded_state['network'])
        last_ckpt_step = loaded_state['training_steps']
        last_ckpt = load_ckpt
        logger.info(f'Evaluator loaded checkpoint "{load_ckpt}"')

    prev_ckpt_network = deepcopy(network).to(device=device)
    disable_auto_grad(prev_ckpt_network)
    network.eval()
    prev_ckpt_network.eval()

    # Load evaluation dataset if available
    dataloader = None
    if eval_games_dir and os.path.exists(eval_games_dir):
        eval_dataset = build_eval_dataset(eval_games_dir, env.num_stack, logger)
        dataloader = DataLoader(eval_dataset, batch_size=1024, pin_memory=True, shuffle=False)

    # Create MCTS players
    black_elo = EloRating(rating=default_rating)
    white_elo = EloRating(rating=default_rating)

    # CRITICAL FIX: Pass use_fx and board_size to evaluator players
    black_player = create_mcts_player(network, device, num_simulations, num_parallel, False, True, use_fx, board_size)
    white_player = create_mcts_player(prev_ckpt_network, device, num_simulations, num_parallel, False, True, use_fx, board_size)

    while not stop_event.is_set():
        ckpt_file = _decode_bytes(var_ckpt.value)
        if not ckpt_file or ckpt_file == last_ckpt or not os.path.exists(ckpt_file):
            time.sleep(30)
            continue

        # Load new checkpoint
        loaded_state = torch.load(ckpt_file, map_location=device)
        training_steps = loaded_state['training_steps']
        network.load_state_dict(loaded_state['network'])
        network.eval()
        last_ckpt = ckpt_file

        # Evaluate
        selfplay_stats = eval_against_prev_ckpt(
            env, black_player, white_player, black_elo, white_elo, c_puct_base, c_puct_init
        )
        pro_game_stats = eval_on_pro_games(network, device, dataloader)

        stats = {
            'datetime': get_time_stamp(),
            'training_steps': training_steps,
            **selfplay_stats,
            **pro_game_stats,
        }
        writer.write(OrderedDict((n, v) for n, v in stats.items()))

        # Save evaluation game
        if save_sgf_dir and os.path.isdir(save_sgf_dir):
            sgf_file = os.path.join(save_sgf_dir, f'eval_{training_steps}_vs_{last_ckpt_step}.sgf')
            with open(sgf_file, 'w') as f:
                f.write(env.to_sgf())

        # Update previous network
        prev_ckpt_network.load_state_dict(loaded_state['network'])
        prev_ckpt_network.eval()
        white_elo = deepcopy(black_elo)
        last_ckpt_step = training_steps

    writer.close()


@torch.no_grad()
def eval_against_prev_ckpt(
    env, black_player, white_player, black_elo, white_elo, c_puct_base, c_puct_init
) -> Mapping[Text, Any]:
    """Play one game between current and previous checkpoint."""
    _ = env.reset()
    done = False
    num_passes = 0

    while not done:
        mcts_player = black_player if env.to_play == env.black_player else white_player
        move, *_ = mcts_player(env, None, c_puct_base, c_puct_init, False)
        _, _, done, _ = env.step(move)

        if env.has_pass_move and move == env.pass_move:
            num_passes += 1

    stats = {
        'game_length': env.steps,
        'game_result': env.get_result_string(),
    }

    if env.has_pass_move:
        stats['num_passes'] = num_passes

    # Update Elo ratings
    if env.winner is not None:
        if env.winner == env.black_player:
            winner, loser = black_elo, white_elo
        else:
            winner, loser = white_elo, black_elo
        winner.update_rating(loser.rating, 1)
        loser.update_rating(winner.rating, 0)

    stats['black_elo_rating'] = black_elo.rating
    stats['white_elo_rating'] = white_elo.rating
    return stats


@torch.no_grad()
def eval_on_pro_games(network, device, dataloader, k_list=(1, 3, 5)) -> Mapping[Text, Any]:
    """Evaluate network accuracy on professional games."""
    if not dataloader:
        return {}

    total_correct = {k: 0 for k in k_list}
    total_entropy = total_mse_loss = total_examples = 0

    for states, target_pi, target_v in dataloader:
        states = states.to(device=device, non_blocking=True)
        target_pi = target_pi.to(device=device, non_blocking=True)
        target_v = target_v.to(device=device, non_blocking=True)

        outputs = network(states)
        if len(outputs) == 4:  # WDLP mode
            pi_logits, v, wdl, plies = outputs
        else:  # Standard mode
            pi_logits, v = outputs
            
        pi = torch.softmax(pi_logits, dim=-1)
        v = v.squeeze(-1)

        batch_size = states.size(0)
        _, pred = torch.topk(pi, max(k_list), dim=1)
        target_indices = torch.argmax(target_pi, dim=1)

        expanded_targets = target_indices.unsqueeze(1).expand(batch_size, max(k_list))
        matches = pred.eq(expanded_targets)
        for k in k_list:
            total_correct[k] += matches[:, :k].any(dim=1).sum().item()

        entropy = -(pi * torch.log(pi + 1e-8)).sum(dim=1)
        total_entropy += entropy.sum().item()
        total_mse_loss += F.mse_loss(v, target_v, reduction='sum').item()
        total_examples += batch_size

    stats = {
        'value_mse_error': total_mse_loss / total_examples,
        'policy_entropy': total_entropy / total_examples,
    }
    for k in k_list:
        stats[f'policy_top_{k}_accuracy'] = total_correct[k] / total_examples

    return stats