import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class TrajectoryGeneratorVariety:
    """Generates a dataset of synthetic trajectories with different types."""

    def __init__(self, num_trajectories_per_type=10, num_points_range=(10, 20), output_dir="."):
        self.num_trajectories_per_type = num_trajectories_per_type
        self.num_points_range = num_points_range
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_linear_trajectory(self, trajectory_id, start_point, num_points, rotation_angle):
        """Generates a single linear trajectory."""
        t = np.linspace(0, 10, num_points)
        x = t
        y = np.zeros(num_points)

        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])

        points = np.vstack([x, y])
        rotated_points = rotation_matrix @ points
        translated_points = rotated_points + start_point[:, np.newaxis]

        return pd.DataFrame({
            'trajectory_id': trajectory_id,
            'timestamp': np.arange(num_points),
            'x': translated_points[0],
            'y': translated_points[1],
            'label': 'linear'
        })

    def _generate_circular_trajectory(self, trajectory_id, start_point, num_points, radius):
        """Generates a single circular trajectory."""
        t = np.linspace(0, 2 * np.pi, num_points)
        x = radius * np.cos(t)
        y = radius * np.sin(t)

        points = np.vstack([x, y])
        translated_points = points + start_point[:, np.newaxis]

        return pd.DataFrame({
            'trajectory_id': trajectory_id,
            'timestamp': np.arange(num_points),
            'x': translated_points[0],
            'y': translated_points[1],
            'label': 'circular'
        })

    def _generate_sinuous_trajectory(self, trajectory_id, start_point, num_points, rotation_angle, freq, amplitude):
        """Generates a single sinuous trajectory."""
        t = np.linspace(0, 4 * np.pi, num_points)
        x = t
        y = amplitude * np.sin(freq * t)

        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])

        points = np.vstack([x, y])
        rotated_points = rotation_matrix @ points
        translated_points = rotated_points + start_point[:, np.newaxis]

        return pd.DataFrame({
            'trajectory_id': trajectory_id,
            'timestamp': np.arange(num_points),
            'x': translated_points[0],
            'y': translated_points[1],
            'label': 'sinuous'
        })

    def generate_dataset(self):
        """Generates the full dataset of trajectories."""
        all_trajectories = []
        trajectory_id_counter = 0

        for _ in range(self.num_trajectories_per_type):
            # Generate linear trajectory
            start_point = np.random.rand(2) * 100
            num_points = np.random.randint(*self.num_points_range)
            rotation_angle = np.random.rand() * 2 * np.pi
            linear_traj = self._generate_linear_trajectory(trajectory_id_counter, start_point, num_points, rotation_angle)
            all_trajectories.append(linear_traj)
            trajectory_id_counter += 1

            # Generate circular trajectory
            start_point = np.random.rand(2) * 100
            num_points = np.random.randint(*self.num_points_range)
            radius = np.random.uniform(5, 15)
            circular_traj = self._generate_circular_trajectory(trajectory_id_counter, start_point, num_points, radius)
            all_trajectories.append(circular_traj)
            trajectory_id_counter += 1

            # Generate sinuous trajectory
            start_point = np.random.rand(2) * 100
            num_points = np.random.randint(*self.num_points_range)
            rotation_angle = np.random.rand() * 2 * np.pi
            freq = np.random.uniform(0.5, 2)
            amplitude = np.random.uniform(5, 15)
            sinuous_traj = self._generate_sinuous_trajectory(trajectory_id_counter, start_point, num_points, rotation_angle, freq, amplitude)
            all_trajectories.append(sinuous_traj)
            trajectory_id_counter += 1
        
        dataset = pd.concat(all_trajectories, ignore_index=True)
        return dataset

    def plot_trajectories(self, df, filename="ex1_traj_orig.png"):
        """Plots the generated trajectories."""
        plt.figure(figsize=(10, 10))
        for label, group in df.groupby('label'):
            for trajectory_id in group['trajectory_id'].unique():
                trajectory = group[group['trajectory_id'] == trajectory_id]
                plt.plot(trajectory['x'], trajectory['y'], '-', label=f'{label} {trajectory_id}')
        
        plt.title('Synthetic Trajectories by Type')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(title='Trajectory Type and ID', loc='best', fontsize=8)
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
