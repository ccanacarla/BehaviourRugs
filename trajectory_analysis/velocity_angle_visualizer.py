import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from trajectory_analysis.extractorMetrics import FeatureExtractor
from matplotlib.colors import LinearSegmentedColormap

class VelocityAngleVisualizer:
    """Generates visualizations based on velocity and angle."""

    def __init__(self, output_dir="."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_speed = None
        self._create_custom_colormap()

    def _create_custom_colormap(self):
        # User-specified colors for cardinal directions
        colors = {
            -180: "#AA0071", # West
            -90:  "#8D30F0", # South
            0:    "#F06B30", # East
            90:   "#F0EF30", # North
            180:  "#AA0071"  # West (to wrap around)
        }
        
        # Normalize angles to [0, 1] range for the colormap
        # Using a sorted list of keys to ensure order
        sorted_angles = sorted(colors.keys())
        nodes = [(angle + 180) / 360.0 for angle in sorted_angles]
        hex_colors = [colors[angle] for angle in sorted_angles]

        self.custom_cmap = LinearSegmentedColormap.from_list(
            "custom_gradient", list(zip(nodes, hex_colors))
        )

    def _speed_angle_to_rgb(self, speed, angle):
        if pd.isna(speed) or pd.isna(angle) or self.max_speed is None:
            return [1.0, 1.0, 1.0]  # White for NaN values

        # 1. Get the base color from the angle using the custom colormap
        angle_normalized_for_cmap = (angle + 180) / 360.0
        base_r, base_g, base_b, _ = self.custom_cmap(angle_normalized_for_cmap)

        # 2. Map speed to a brightness factor (from 0 to 1)
        brightness_factor = min(speed / self.max_speed, 1.0) if self.max_speed > 0 else 0
        
        # 3. Apply brightness to the base RGB color by scaling each component
        final_rgb = (base_r * brightness_factor, 
                     base_g * brightness_factor, 
                     base_b * brightness_factor)

        return final_rgb

    def plot_velocity_angle_rugs(self, df, filename="velocity_angle_rugs.png"):
        """
        Creates a 'rug' plot where color represents velocity and angle.
        - Color is determined by the angle of movement.
        - Brightness is determined by the speed.
        """
        # --- Garantir coordenadas ---
        if 'x_rotated' not in df.columns or 'y_rotated' not in df.columns:
            df['x_rotated'] = df['x']
            df['y_rotated'] = df['y']

        feature_extractor = FeatureExtractor(df.copy())
        featured_df = feature_extractor.extract_features()

        # --- Normalizar velocidade ---
        self.max_speed = featured_df["speed"].quantile(0.99) if not featured_df.empty else 1

        trajectories = featured_df.groupby("trajectory_id")
        if trajectories.ngroups == 0:
            plt.figure(figsize=(8, 3))
            plt.text(0.5, 0.5, "No trajectories to display", ha="center", va="center")
            plt.axis("off")
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight", pad_inches=0.05)
            plt.close()
            return

        max_len = int(trajectories.size().max())
        n_traj = int(trajectories.ngroups)

        # --- Construir matriz de cores ---
        image = np.zeros((n_traj, max_len, 3))
        for i, (traj_id, traj_df) in enumerate(trajectories):
            colors = [self._speed_angle_to_rgb(r["speed"], r["angle_deg"]) for _, r in traj_df.iterrows()]
            if len(colors) < max_len:
                colors += [[0, 0, 0]] * (max_len - len(colors))
            image[i] = np.array(colors[:max_len])

        # --- Criar plot ---
        plt.figure(figsize=(12, max(n_traj * 0.3, 3)))
        plt.imshow(image, aspect='auto', interpolation='nearest')

        # --- Y ticks: trajectory_id ---
        traj_labels = list(trajectories.groups.keys())
        max_label_len = 30
        short_labels = [str(l) if len(str(l)) <= max_label_len else str(l)[:max_label_len-3] + "..." for l in traj_labels]
        plt.yticks(np.arange(n_traj), short_labels)

        # --- X ticks mais divididos + rotação ---
        if max_len <= 1000:
            step = 100
        elif max_len <= 5000:
            step = 500
        elif max_len <= 20000:
            step = 1000
        else:
            step = max_len // 20

        xticks = np.arange(0, max_len + 1, 200)
        plt.xticks(xticks, rotation=45)
        plt.xlabel("Time (sequence index)")
        plt.ylabel("Trajectories")

        plt.title("Velocity and Angle Representation of Trajectories", pad=14, fontsize=12)

        # --- Grade leve para facilitar leitura ---
        plt.grid(axis='x', color='gray', linestyle='--', linewidth=0.4, alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight', pad_inches=0.04)
        plt.close()


    def plot_legend(self, filename="velocity_angle_legend.png"):
        """
        Creates a legend image that maps angle and speed to colors.
        """
        if self.max_speed is None:
            if not hasattr(self, 'custom_cmap'):
                self._create_custom_colormap()
            print("Warning: max_speed is not set. Using a default of 1.0 for the legend.")
            self.max_speed = 1.0


        angle_range = np.linspace(-180, 180, 360)
        speed_range = np.linspace(0, self.max_speed, 100)
        
        legend_image = np.zeros((100, 360, 3))
        
        for i, speed in enumerate(speed_range):
            for j, angle in enumerate(angle_range):
                legend_image[99-i, j] = self._speed_angle_to_rgb(speed, angle)

        plt.figure(figsize=(10, 6)) # Increased height
        plt.imshow(legend_image, aspect='auto')
        plt.title("Velocity-Angle Color Legend")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Speed")
        
        plt.xticks(ticks=[0, 90, 180, 270, 359], labels=[-180, -90, 0, 90, 180])
        plt.yticks(ticks=[0, 50, 99], labels=[f'{self.max_speed:.2f}', f'{self.max_speed/2:.2f}', '0.00'])

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()
