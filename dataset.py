from typing import Tuple, List
from simulator import ProjectileSimulator, Method
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Tensor
import torch


class ProjectileDatasetGenerator():
    # constants
    dtype: torch.dtype = torch.float32

    def __init__(
            self,
            angle_range: Tuple[float, float, float],
            delta_time: float,
            max_distance: float
    ) -> None:
        # attributes
        self.angles = angle_range

        # simulator instance
        self.simulator = ProjectileSimulator(
            delta_time=delta_time,
            max_distance=max_distance
        )
        self.gravity_y = self.simulator.gravity[1]

        # progress bar
        self.progress = Progress(
            TextColumn('[progress.description]{task.description}{task.fields[angle]}'),
            BarColumn(),
            TextColumn('{task.percentage:>3.2f}%'),
            TimeRemainingColumn()
        )

    def quadratic_formula(self, a: float, b: float, c: float, sign: int) -> float:
        sqrt = b**2 - 4*a*c
        sqrt = torch.clamp(sqrt, min=0)
        return (-b + sign*torch.sqrt(sqrt))/(2*a)
    
    def generate_dataset(
            self,
            muzzle_velocity: float,
            drag_coefficient: float,
            drag_divisor: float,
            method: Method
    ) -> List:
        # add progress bar
        task = self.progress.add_task(
            description='Generating dataset...',
            total=len(self.angles),
            angle=''
        )
        self.progress.start()

        # dataset list
        dataset = []

        # iterate over angles
        for angle in self.angles:
            # simulate projectile motion
            self.simulator.build_state(muzzle_velocity, angle)
            self.simulator.simulate(drag_coefficient, method)

            # convert muzzle velocity from m/s to inch/s
            inch_velocity = self.simulator.meter_to_inch*muzzle_velocity

            # distinguish between high and low angles
            arc = -1 if angle < 45 else 1

            for state in self.simulator.state:
                x = state.position[0]
                y = state.position[1]

                # approximate launch angle
                temp = self.gravity_y*x**2/(2*inch_velocity**2)
                launch_angle = self.quadratic_formula(
                    a=temp,
                    b=x,
                    c=temp - y,
                    sign=arc
                )
                launch_angle = torch.atan(launch_angle)

                # approximate flight time
                v_cos = inch_velocity*torch.cos(launch_angle)
                flight_time = self.quadratic_formula(
                    a=(drag_coefficient*v_cos**2)/drag_divisor,
                    b=v_cos,
                    c=torch.sqrt(x**2 + y**2)*torch.cos(torch.atan(y/x)),
                    sign=1
                )

                # check for nan values
                if launch_angle.isnan() or flight_time.isnan():
                    continue

                # final data processing
                launch_angle = -torch.rad2deg(launch_angle)
                flight_time = -flight_time

                # concat dataset
                dataset.append(torch.cat([
                    torch.tensor([
                        muzzle_velocity,
                        drag_coefficient,
                        arc,
                        x/self.simulator.max_distance_x,
                        y/self.simulator.max_distance_y,
                        launch_angle,
                        flight_time
                    ], dtype=self.dtype).unsqueeze(0),

                    torch.tensor([
                        angle,
                        state.time
                    ], dtype=self.dtype).unsqueeze(0)
                ], dim=1))

            # update progress bar
            self.progress.update(task, advance=1, angle=f'[magenta]{angle:.2f}°')

        # remove progress bar
        self.progress.stop()
        return dataset


class ProjectileDataset(Dataset):
    def __init__(
            self,
            dataset_generator: ProjectileDatasetGenerator,
            muzzle_velocity: float,
            drag_coefficient: float,
            drag_divisor: float,
            method: Method
    ) -> None:
        self.dataset = dataset_generator.generate_dataset(muzzle_velocity, drag_coefficient, drag_divisor, method)

        # concat after generating dataset
        self.dataset = torch.cat(self.dataset)

    def __len__(self) -> int:
        return self.dataset.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        inputs = self.dataset[index, :-2] # muzzle_velocity, drag_coefficient, arc, x, y, launch_angle, flight_time
        labels = self.dataset[index, -2:] # angle, time
        
        # convert to tensor if not already
        if not isinstance(inputs, Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        if not isinstance(labels, Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)
        return inputs, labels


class ProjectileDataLoader(DataLoader):
    def __init__(
            self,
            dataset: ProjectileDataset,
            split: Tuple[float, float] = (0.8, 0.1),
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            pin_memory: bool = False
    ) -> None:
        # attributes
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def get_dataloader(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # split dataset
        total_size = len(self.dataset)
        train_size = int(self.split[0]*total_size)
        val_size = int(self.split[1]*total_size)
        test_size = total_size - train_size - val_size
        print(f"train_size: {train_size}, val_size: {val_size}, test_size: {test_size}" )

        # random split
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset,
            [train_size, val_size, test_size]
        )

        # dataloaders
        args = {
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory
        }
        return (
            DataLoader(train_dataset, **args),
            DataLoader(val_dataset, **args),
            DataLoader(test_dataset, **args)
        )

