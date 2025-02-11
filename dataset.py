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
        self.angles = torch.linspace(*angle_range, dtype=self.dtype)

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

    def quadratic(self, a: float, b: float, c: float, sign: int) -> float:
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
        # initialize progress bar
        task = self.progress.add_task(
            description='Generating dataset...',
            total=len(self.angles),
            angle=''
        )
        self.progress.start()

        # dataset list
        dataset = []

        for angle in self.angles:
            # simulate projectile motion
            self.simulator.build_state(muzzle_velocity, angle)
            self.simulator.simulate(drag_coefficient, method)

            # convert muzzle_velocity from m/s to inch/s
            inch_velocity = self.simulator.meter_to_inch*muzzle_velocity

            # distinguish between high and low angles
            arc = -1 if angle < 45 else 1

            for state in self.simulator.state:
                x = state.position[0]
                y = state.position[1]

                # approximate launch angle
                tmp = self.gravity_y*x**2/(2*inch_velocity**2)
                launch_angle = self.quadratic(
                    a=tmp,
                    b=x,
                    c=tmp-y,
                    sign=arc
                )
                launch_angle = torch.atan(launch_angle)

                # approximate flight time
                v_cos = inch_velocity*torch.cos(launch_angle)
                flight_time = self.quadratic(
                    a=(drag_coefficient*v_cos**2)/drag_divisor,
                    b=v_cos,
                    c=torch.sqrt(x**2 + y**2)*torch.cos(torch.atan(y/x)),
                    sign=1
                )

                # final data processing
                launch_angle = -torch.rad2deg(launch_angle)
                flight_time = -flight_time

                # concat the state vectors to tensor
                features = torch.cat([
                    # model inputs: x, y, arc, launch_angle, flight_time            (1, 5)
                    torch.tensor([x, y], dtype=self.dtype).unsqueeze(0),          # (1, 2)
                    torch.tensor([arc], dtype=self.dtype).unsqueeze(0),           # (1, 1)
                    torch.tensor([launch_angle], dtype=self.dtype).unsqueeze(0),  # (1, 1)
                    torch.tensor([flight_time], dtype=self.dtype).unsqueeze(0),   # (1, 1)

                    # model outputs: angle, time                                    (1, 2)
                    torch.tensor([angle], dtype=self.dtype).unsqueeze(0),         # (1, 1)
                    torch.tensor([state.time], dtype=self.dtype).unsqueeze(0),    # (1, 1)

                ], dim=1)
                dataset.append(features)

            # update progress bar
            self.progress.update(task, advance=1, angle=f'[magenta]{angle:.2f}°')
        # stop progress bar
        self.progress.stop()
        return dataset


class ProjectileDataset(Dataset):
    # constants
    dtype: torch.dtype = torch.float32

    def __init__(
            self,
            angle_range: Tuple[float, float, float],
            delta_time: float,
            max_distance: float,
            muzzle_velocity: float,
            drag_coefficient: float,
            drag_divisor: float,
            method: Method
    ) -> None:
        # generate dataset
        dataset = ProjectileDatasetGenerator(
            angle_range=angle_range,
            delta_time=delta_time,
            max_distance=max_distance
        ).generate_dataset(
            muzzle_velocity=muzzle_velocity,
            drag_coefficient=drag_coefficient,
            drag_divisor=drag_divisor,
            method=method
        )

        self.dataset = torch.cat(dataset, dim=0)

    def __len__(self) -> int:
        return self.dataset.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # model inputs: x, y, arc, launch_angle, flight time
        inputs = self.dataset[index, :5]

        # model outputs: angle, time
        outputs = self.dataset[index, 5:]
        return inputs, outputs


class ProjectileDataLoader(DataLoader):
    default_split: Tuple[float, float] = (0.8, 0.1)

    def __init__(
            self,
            dataset: ProjectileDataset,
            dataset_split: Tuple[float, float] = default_split,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            pin_memory: bool = False
    ) -> None:
        self.dataset = dataset
        self.dataset_split = dataset_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def get_dataloader(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # split the dataset
        total_size = len(self.dataset)
        train_size = int(total_size*self.dataset_split[0])
        val_size = int(total_size*self.dataset_split[1])
        test_size = total_size - train_size - val_size

        # get the dataloaders
        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        
        loader = []
        for phase in [train_dataset, val_dataset, test_dataset]:
            loader.append(
                DataLoader(
                    dataset=phase,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory
                )
            )
        print(f'Train: {train_size}, Validation: {val_size}, Test: {test_size}')
        return loader

