# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Replay components for training agents."""

from typing import Mapping, Text, Any, NamedTuple, Optional, Sequence
import numpy as np
import snappy


class Transition(NamedTuple):
    """
    Transition tuple for storing game experiences.
    
    INNOVATION: Extended to support WDLP (Win-Draw-Loss-Plies) targets.
    Backward compatible: wdl and plies are optional.
    """
    state: Optional[np.ndarray]
    pi_prob: Optional[np.ndarray]
    value: Optional[float]
    wdl: Optional[np.ndarray] = None   # INNOVATION: Win-Draw-Loss target [3]
    plies: Optional[float] = None      # INNOVATION: Remaining plies target


# Default structure for backward compatibility
TransitionStructure = Transition(state=None, pi_prob=None, value=None, wdl=None, plies=None)


def compress_array(array):
    """Compresses a numpy array with snappy."""
    return snappy.compress(array), array.shape, array.dtype


def uncompress_array(compressed):
    """Uncompresses a numpy array with snappy given its shape and dtype."""
    compressed_array, shape, dtype = compressed
    byte_string = snappy.uncompress(compressed_array)
    return np.frombuffer(byte_string, dtype=dtype).reshape(shape)


def value_to_wdl(value: float) -> np.ndarray:
    """
    Convert scalar value to WDL (Win-Draw-Loss) distribution.
    
    INNOVATION: Mapping strategy for Gomoku (no draws in standard rules).
    Based on "Representation Matters for Mastering Chess" (Czech et al., 2024)
    
    Args:
        value: Position evaluation in [-1, +1]
            +1 = player 1 wins
            -1 = player 1 loses
            0 = uncertain/balanced
            
    Returns:
        wdl: [P(win), P(draw), P(loss)] probabilities
        
    For Gomoku (no draws in standard rules):
        - Use draw channel for uncertainty
        - Sharp conversion near ±1 (certain outcomes)
        - Smooth transition in middle range (uncertain)
    """
    wdl = np.zeros(3, dtype=np.float32)
    
    if value > 0.8:  # Clear win
        wdl[0] = 1.0  # Win
        wdl[1] = 0.0  # Draw
        wdl[2] = 0.0  # Loss
    elif value < -0.8:  # Clear loss
        wdl[0] = 0.0  # Win
        wdl[1] = 0.0  # Draw
        wdl[2] = 1.0  # Loss
    else:  # Uncertain - use draw channel
        # Smooth interpolation
        win_prob = (value + 1.0) / 2.0  # Map [-1, 1] → [0, 1]
        loss_prob = 1.0 - win_prob
        
        # Uncertainty factor (highest at value=0)
        uncertainty = 1.0 - abs(value)
        draw_prob = uncertainty * 0.5  # Max 50% uncertainty at value=0
        
        # Renormalize
        win_prob *= (1.0 - draw_prob)
        loss_prob *= (1.0 - draw_prob)
        
        wdl[0] = win_prob
        wdl[1] = draw_prob
        wdl[2] = loss_prob
    
    return wdl


class UniformReplay:
    """
    Uniform replay, with circular buffer storage for flat named tuples.
    
    INNOVATION: Extended to support WDLP (Win-Draw-Loss-Plies) targets
    for enhanced value head training.
    """

    def __init__(
        self,
        capacity: int,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        compress_data: bool = True,
        use_wdlp: bool = False,  # INNOVATION: Enable WDLP support
    ):
        if capacity <= 0:
            raise ValueError(f'Expect capacity to be a positive integer, got {capacity}')
        self.structure = TransitionStructure
        self.capacity = capacity
        self.random_state = random_state
        self.compress_data = compress_data
        self.use_wdlp = use_wdlp  # Track if using WDLP
        self.storage = [None] * capacity

        self.num_games_added = 0
        self.num_samples_added = 0

    def add_game(self, game_seq: Sequence[Transition]) -> None:
        """Add an entire game to replay."""

        for transition in game_seq:
            self.add(transition)

        self.num_games_added += 1

    def add(self, transition: Any) -> None:
        """Adds single transition to replay."""
        index = self.num_samples_added % self.capacity
        self.storage[index] = self.encoder(transition)
        self.num_samples_added += 1

    def get(self, indices: Sequence[int]) -> Sequence[Transition]:
        """Retrieves items by indices."""
        return [self.decoder(self.storage[i]) for i in indices]

    def sample(self, batch_size: int) -> Transition:
        """Samples batch of items from replay uniformly, with replacement."""
        if self.size < batch_size:
            return

        indices = self.random_state.randint(low=0, high=self.size, size=batch_size)
        samples = self.get(indices)

        transposed = zip(*samples)
        stacked = []
        for xs in transposed:
            # Handle None values (for backward compatibility)
            if xs[0] is not None:
                stacked.append(np.stack(xs, axis=0))
            else:
                stacked.append(None)
        
        return type(self.structure)(*stacked)

    def encoder(self, transition: Transition) -> Transition:
        """Encode transition for storage."""
        if self.compress_data:
            encoded = transition._replace(state=compress_array(transition.state))
            
            # INNOVATION: Compress WDL if present
            if self.use_wdlp and transition.wdl is not None:
                encoded = encoded._replace(wdl=compress_array(transition.wdl))
            
            return encoded
        return transition

    def decoder(self, transition: Transition) -> Transition:
        """Decode transition from storage."""
        if self.compress_data:
            decoded = transition._replace(state=uncompress_array(transition.state))
            
            # INNOVATION: Decompress WDL if present
            if self.use_wdlp and transition.wdl is not None:
                decoded = decoded._replace(wdl=uncompress_array(transition.wdl))
            
            return decoded
        return transition

    def get_state(self) -> Mapping[Text, Any]:
        """Retrieves replay state as a dictionary (e.g. for serialization)."""
        return {
            'num_games_added': self.num_games_added,
            'num_samples_added': self.num_samples_added,
            'storage': self.storage,
            'use_wdlp': self.use_wdlp,  # INNOVATION: Save WDLP flag
        }

    def set_state(self, state: Mapping[Text, Any]) -> None:
        """Sets replay state from a (potentially de-serialized) dictionary."""
        self.num_games_added = state['num_games_added']
        self.num_samples_added = state['num_samples_added']
        self.storage = state['storage']
        self.use_wdlp = state.get('use_wdlp', False)  # Backward compatibility

    @property
    def size(self) -> int:
        """Number of items currently contained in replay."""
        return min(self.num_samples_added, self.capacity)