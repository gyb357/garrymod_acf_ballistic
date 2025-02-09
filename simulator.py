from dataclasses import dataclass
from torch import Tensor
from typing import Callable, Tuple, List
import torch
import math


@dataclass
class StateVectors():
    position: Tensor  # (2,) [x, y]
    velocity: Tensor  # (2,) [vx, vy]
    time: Tensor      # (1,) [t]


# defines the format of the parameters for numerical analysis methods
Method = Callable[[Tensor, Tensor, Tensor, float, Tensor, float], Tuple[Tensor, Tensor, Tensor]]


class ProjectileSimulator():
    # constants
    dtype: torch.dtype = torch.float32
    meter_to_inch: float = 39.3701                            # 1 meter = 39.3701 inch
    gravity: Tensor = torch.tensor([0, -600.0], dtype=dtype)  # inch/s^2

    def __init__(self, delta_time: float, max_distance: float) -> None:
        # attributes
        self.delta_time = delta_time
        self.max_distance_x = math.sqrt(3*max_distance**2)
        self.max_distance_y = -max_distance

        # initial state vectors
        self.state: List[StateVectors] = []

    def build_state(self, velocity: float, angle: float) -> None:
        inch_velocity = self.meter_to_inch*velocity
        rad_angle = math.radians(angle)

        # initial position
        self.state = [
            StateVectors(
                position=torch.tensor([0, 0], dtype=self.dtype),
                velocity=torch.tensor([inch_velocity*math.cos(rad_angle),
                                       inch_velocity*math.sin(rad_angle)], dtype=self.dtype),
                time=torch.tensor([0], dtype=torch.float32)
            )
        ]

    def simulate(self, drag_coefficient: float, method: Method) -> None:
        while True:
            current_state = self.state[-1]

            # get the next state
            p_next, v_next, t_next = method(
                current_state.position,
                current_state.velocity,
                current_state.time,
                drag_coefficient,
                self.gravity,
                self.delta_time
            )

            # check if the projectile is out of bounds
            if p_next[0].item() > self.max_distance_x or \
               p_next[1].item() < self.max_distance_y:
                break

            # append the next state
            self.state.append(StateVectors(p_next, v_next, t_next))


class NumericalAnalysis():
    @staticmethod
    def euler(
        p_current: Tensor,
        v_current: Tensor,
        t_current: Tensor,
        drag_coefficient: float,
        gravity: Tensor,
        delta_time: float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        drag = torch.norm(v_current)*drag_coefficient
        accel = (gravity - drag)*delta_time

        # update state vectors
        p_next = p_current + (v_current + 0.5*accel)*delta_time
        v_next = v_current + accel
        t_next = t_current + delta_time
        return p_next, v_next, t_next
    
    @staticmethod
    def rk2(
        p_current: Tensor,
        v_current: Tensor,
        t_current: Tensor,
        drag_coefficient: float,
        gravity: Tensor,
        delta_time: float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        drag = torch.norm(v_current)*drag_coefficient

        v1 = v_current
        a1 = gravity - v1*drag
        v2 = v1 + a1*delta_time
        a2 = gravity - v2*drag

        # update state vectors
        p_next = p_current + (delta_time/2)*(v1 + v2)
        v_next = v_current + (delta_time/2)*(a1 + a2)
        t_next = t_current + delta_time
        return p_next, v_next, t_next
    
    @staticmethod
    def rk4(
        p_current: Tensor,
        v_current: Tensor,
        t_current: Tensor,
        drag_coefficient: float,
        gravity: Tensor,
        delta_time: float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        drag = torch.norm(v_current)*drag_coefficient

        v1 = v_current
        a1 = gravity - v1*drag
        v2 = v1 + a1*(delta_time/2)
        a2 = gravity - v2*drag
        v3 = v1 + a2*(delta_time/2)
        a3 = gravity - v3*drag
        v4 = v1 + a3*delta_time
        a4 = gravity - v4*drag

        # update state vectors
        p_next = p_current + (delta_time/6)*(v1 + 2*(v2 + v3) + v4)
        v_next = v_current + (delta_time/6)*(a1 + 2*(a2 + a3) + a4)
        t_next = t_current + delta_time
        return p_next, v_next, t_next

