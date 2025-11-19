import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class TrajectoryPreprocessor:
    """Translates and rotates trajectories to a canonical form."""

    def __init__(self, df, output_dir="."):
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.pca_directions = {}

    def _angle_from_pca(self, x, y):
        """Calculates the principal component angle."""
        if len(x) < 2:
            return 0.0, (0, 0), np.array([1, 0])
        X = np.vstack([x, y]).T
        Xc = X - np.mean(X, axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        v1 = Vt[0]

        # Resolve ambiguity: ensure the eigenvector points in the same
        # general direction as the trajectory's displacement.
        # The trajectory is already translated to start at (0,0).
        displacement_vector = np.array([x[-1], y[-1]])
        if np.dot(v1, displacement_vector) < 0:
            v1 = -v1

        theta = np.arctan2(v1[1], v1[0])
        return theta, np.mean(X, axis=0), v1

    def _rotate_points_about_origin(self, x, y, theta):
        """Rotates points by a given angle theta."""
        c = np.cos(theta)
        s = np.sin(theta)
        x_rot = x * c - y * s
        y_rot = x * s + y * c
        return x_rot, y_rot

    def process(self):
        """Runs the full preprocessing pipeline."""
        # 1. Translate to origin
        self.df['x_translate'] = self.df.groupby('trajectory_id')['x'].transform(lambda x: x - x.iloc[0])
        self.df['y_translate'] = self.df.groupby('trajectory_id')['y'].transform(lambda y: y - y.iloc[0])

        # 2. Rotate by PCA
        self.df['x_rotated'] = np.nan
        self.df['y_rotated'] = np.nan
        self.df['theta_pca'] = np.nan

        for tid, g in self.df.groupby('trajectory_id'):
            x = g['x_translate'].to_numpy()
            y = g['y_translate'].to_numpy()
            if len(x) == 0:
                continue

            theta_pca, center, v1 = self._angle_from_pca(x, y)
            x_rot, y_rot = self._rotate_points_about_origin(x, y, -theta_pca)

            self.df.loc[g.index, 'x_rotated'] = x_rot
            self.df.loc[g.index, 'y_rotated'] = y_rot
            self.df.loc[g.index, 'theta_pca'] = theta_pca
            self.pca_directions[tid] = {'center': center, 'vector': v1, 'theta': theta_pca}
        
        print(self.df.head(10))
        return self.df

    def plot_comparison(self, filename="ex1_Comparacao_Original_Rotacionado_PCA.png"):
        """Plots original vs. rotated trajectories, grouped by trajectory_id."""

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # --- AGRUPA POR TRAJECTORY_ID UMA ÚNICA VEZ ---
        grouped = self.df.groupby("trajectory_id")
        traj_ids = list(grouped.groups.keys())

        # Paleta de cores consistente para os dois painéis
        colors = plt.cm.tab20(np.linspace(0, 1, len(traj_ids)))
        color_map = {tid: c for tid, c in zip(traj_ids, colors)}

        # -------- Painel 1: Espaço original (transladado) + direções PCA --------
        ax = axes[0]

        for traj_id, data_id in grouped:
            data_id = data_id.sort_values("timestamp")
            color = color_map[traj_id]

            # Trajetória original
            ax.plot(
                data_id["x_translate"],
                data_id["y_translate"],
                label=f"Traj {traj_id}",
                color=color,
                linewidth=2,
                alpha=0.9
            )

            # Direção PCA (se existir)
            if traj_id in self.pca_directions:
                center = np.array(self.pca_directions[traj_id]['center'])
                v1 = np.array(self.pca_directions[traj_id]['vector'])

                scale = np.max(
                    np.abs(
                        data_id[["x_translate", "y_translate"]].to_numpy()
                    )
                ) * 0.5

                line_x = [center[0] - v1[0]*scale, center[0] + v1[0]*scale]
                line_y = [center[1] - v1[1]*scale, center[1] + v1[1]*scale]

                ax.plot(line_x, line_y, '--', color='gray', linewidth=1, alpha=0.6)

        ax.set_title("Original Space (Translated) + PCA Directions")
        ax.set_xlabel("X (translated)")
        ax.set_ylabel("Y (translated)")
        ax.axis('equal')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(fontsize=8)

        # -------- Painel 2: Trajetórias rotacionadas --------
        ax = axes[1]

        for traj_id, data_id in grouped:
            data_id = data_id.sort_values("timestamp")
            color = color_map[traj_id]

            ax.plot(
                data_id["x_rotated"],
                data_id["y_rotated"],
                label=f"Traj {traj_id}",
                color=color,
                linewidth=2,
                alpha=0.9
            )

        ax.set_title("Rotated Space (Aligned to X-axis)")
        ax.set_xlabel("X Rotated")
        ax.set_ylabel("Y Rotated")
        ax.axis('equal')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(fontsize=8)

        plt.suptitle("Comparison: Before and After PCA Rotation", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
