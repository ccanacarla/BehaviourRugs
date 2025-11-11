import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

class Visualizer:
    """Generates visualizations for trajectory analysis."""

    def __init__(self, output_dir="."):
        self.output_dir = Path(output_dir)

    def plot_behaviour_rugs(self, symbolic_df, filename="ex1_BehaviourRugs.png"):
        """Creates a 'behaviour rug' plot from symbolic data."""
        direction_rgb = {
            'Norte': [1.0, 0.0, 0.0], 'Nordeste': [1.0, 0.5, 0.0], 'Leste': [1.0, 1.0, 0.0],
            'Sudeste': [0.0, 1.0, 0.0], 'Sul': [0.0, 0.0, 1.0], 'Sudoeste': [0.5, 0.0, 0.5],
            'Oeste': [1.0, 0.0, 1.0], 'Noroeste': [1.0, 0.75, 0.8]
        }
        speed_factor = {'Lento': 0.4, 'Médio': 0.7, 'Rápido': 1.0}

        def symbol_to_rgb(symbol):
            if pd.isna(symbol) or symbol == "": return [1.0, 1.0, 1.0]
            for speed in speed_factor:
                if symbol.startswith(speed):
                    direction = symbol[len(speed):]
                    base_color = direction_rgb.get(direction, [0, 0, 0])
                    brightness = speed_factor[speed]
                    return np.clip(np.array(base_color) * brightness, 0, 1)
            return [0, 0, 0]

        max_len = max(len(seq) for seq in symbolic_df["simbolic_movement"])
        n_traj = len(symbolic_df)
        image = np.zeros((n_traj, max_len, 3))

        df_sorted = symbolic_df.sort_values("cluster_id")

        for i, seq in enumerate(df_sorted["simbolic_movement"]):
            colors = [symbol_to_rgb(s) for s in seq]
            if len(colors) < max_len:
                colors += [[0, 0, 0]] * (max_len - len(colors))
            image[i] = np.array(colors[:max_len])

        plt.figure(figsize=(12, n_traj * 0.3))
        plt.imshow(image, aspect='auto')
        
        yticks = np.arange(n_traj)
        ylabels = df_sorted["group"].astype(str) + " (C:" + df_sorted["cluster_id"].astype(str) + ")"
        plt.yticks(yticks, ylabels)

        plt.xlabel("Time (state sequence)")
        plt.ylabel("Trajectories (Group ID & Cluster ID)")
        plt.title("Symbolic Representation of Trajectories")

        legend_elements = [
            mpatches.Patch(
                color=np.clip(np.array(direction_rgb[direction]) * speed_factor[speed], 0, 1),
                label=f"{speed}{direction}"
            )
            for direction in direction_rgb for speed in speed_factor
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_motif_frequency(self, motif_df, filename="motif_frequency.png"):
        """
        Plots the motif frequency vectors for each trajectory as a heatmap.

        Args:
            motif_df (pd.DataFrame): A DataFrame where rows are trajectories and 
                                     columns are motifs, with values representing
                                     the frequency of each motif.
            filename (str): The name of the file to save the plot.
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(motif_df, cmap='viridis', aspect='auto')
        
        plt.colorbar(label='Motif Frequency')
        plt.title('Motif Frequency per Trajectory')
        plt.xlabel('Motifs')
        plt.ylabel('Trajectories')
        
        plt.xticks(ticks=np.arange(len(motif_df.columns)), labels=motif_df.columns, rotation=90)
        plt.yticks(ticks=np.arange(len(motif_df.index)), labels=motif_df.index)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()