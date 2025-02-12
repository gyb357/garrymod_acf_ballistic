from torch import device
from typing import Tuple
from simulator import Method

import torch.nn as nn

class Trainer():
    def __init__(
            self,
            # device parameters
            device: device,
            model: nn.Module,
            # training parameters
            criterion: nn.Module,
            epochs: int,
            accumulate_steps: int,
            checkpoint_steps: int,
            lf: float,
            # dataset parameters
            muzzle_velocity_range: Tuple[float, float, float],
            drag_coefficient_range: Tuple[float, float, float],
            angle_range: Tuple[float, float, float],
            delta_time: float,
            max_distance: float,
            drag_divisor: float,
            method: Method,
            # dataset loader parameters
            dataset_split: Tuple[float, float, float],
            batch_size: int,
            shuffle: bool,
            num_workers: int,
            pin_memory: bool
    ) -> None:
        pass