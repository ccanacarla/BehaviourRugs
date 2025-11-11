import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class FeatureExtractor:
    """Calculates speed and angle for trajectories."""

    def __init__(self, df, window=1, dt=1.0, output_dir="."):
        self.df = df.sort_values(["trajectory_id", "timestamp"]).reset_index(drop=True)
        self.window = int(window)
        self.dt = dt
        self.output_dir = Path(output_dir)
        if not (1 <= self.window <= 30):
            raise ValueError("Window must be between 1 and 30.")

    def _add_speed_angle_to_group(self, g):
        denom = max(self.window - 1, 1)
        dx = g["x_rotated"] - g["x_rotated"].shift(denom)
        dy = g["y_rotated"] - g["y_rotated"].shift(denom)
        
        g["speed"] = np.hypot(dx, dy) / (denom * self.dt)
        g["angle_deg"] = np.degrees(np.arctan2(dy, dx))
        g["window"] = self.window
        g["dt_seconds"] = self.dt
        return g

    def extract_features(self):
        """Applies feature extraction to the whole dataframe."""
        self.df = self.df.groupby("trajectory_id", group_keys=False).apply(self._add_speed_angle_to_group)
        return self.df

    def plot_series(self, column, title, ylabel, filename):
        """Plots a time series for a given column."""
        plt.figure(figsize=(16, 8))
        for id_tracker in self.df['trajectory_id'].unique():
            data_id = self.df[self.df["trajectory_id"] == id_tracker]
            plt.plot(data_id["timestamp"], data_id[column], label=f"Tracker ID: {id_tracker}")
        
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel(ylabel)
        plt.legend(title='Trajectory ID', loc='best', fontsize=8)
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()