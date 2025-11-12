import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class TrajectoryGenerator:
    """Generates a dataset of synthetic sinuous trajectories."""

    def __init__(self, num_trajectories=10, num_points_range=(10, 20), freq_range=(1, 1), amp_range=(1, 1), output_dir="."):
        self.num_trajectories = num_trajectories
        self.num_points_range = num_points_range
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_single_trajectory(self, trajectory_id, start_point, num_points, rotation_angle, freq, amplitude):
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
            'y': translated_points[1]
        })

    def generate_dataset(self):
        """Generates the full dataset of trajectories."""
        all_trajectories = []
        for i in range(self.num_trajectories):
            start_point = np.random.rand(2) * 100
            num_points = np.random.randint(*self.num_points_range)
            rotation_angle = np.random.rand() * 2 * np.pi
            freq = np.random.uniform(*self.freq_range)
            amplitude = np.random.uniform(*self.amp_range)

            trajectory_df = self._generate_single_trajectory(
                trajectory_id=i,
                start_point=start_point,
                num_points=num_points,
                rotation_angle=rotation_angle,
                freq=freq,
                amplitude=amplitude
            )
            all_trajectories.append(trajectory_df)
        
        dataset = pd.concat(all_trajectories, ignore_index=True)
        return dataset

    def plot_trajectories(self, df, filename="ex1_traj_orig.png"):
        """Plots the generated trajectories."""
        plt.figure(figsize=(10, 10))
        for trajectory_id in df['trajectory_id'].unique():
            trajectory = df[df['trajectory_id'] == trajectory_id]
            plt.plot(trajectory['x'], trajectory['y'], '-', label=f'Trajectory {trajectory_id}')
        
        plt.title('Synthetic Sinuous Trajectories')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(title='Trajectory ID', loc='best', fontsize=8)
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
