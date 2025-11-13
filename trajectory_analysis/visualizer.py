import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import shutil

class Visualizer:
    """Generates visualizations for trajectory analysis."""

    def __init__(self, output_dir="."):
        self.output_dir = Path(output_dir)

    def plot_behaviour_rugs(self, symbolic_df, filename="ex1_BehaviourRugs.png"):
        """Creates a 'behaviour rug' plot from symbolic data."""
        direction_rgb = {
            'Norte': [240/255, 239/255, 48/255],   # #F0EF30
            'Leste': [240/255, 107/255, 48/255],   #F06B30
            'Sul': [141/255, 48/255, 240/255],     #8D30F0
            'Oeste': [48/255, 240/255, 217/255]     # #AA0071
        }
        '''
        direction_rgb = {
            'Norte': [56/255, 159/255, 255/255],   # #389FFF
            'Leste': [139/255, 54/255, 245/255],
            'Sul': [230/255, 38/255, 244/255],
            'Oeste': [54/255, 213/255, 245/255]     # #AA0071
        }'''
        speed_factor = {'Parado': 0.01, 'Lento': 0.4, 'Médio': 0.7, 'Rápido': 1.0}

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
        plt.imshow(image, aspect='auto', interpolation='nearest')
        
        yticks = np.arange(n_traj)
        ylabels = df_sorted["group"].astype(str) + " (C:" + df_sorted["cluster_id"].astype(str) + ")"
        plt.yticks(yticks, ylabels)

        step = max_len // 10 if max_len > 1000 else 100  # ajuste dinâmico
        if step < 50:
            step = 50  # limite inferior
        xticks = np.arange(0, max_len + 1, 200)
        plt.xticks(xticks)
        plt.xticks(rotation=45)
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
        plt.figure(figsize=(len(motif_df.columns)*0.3, 10))
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

    def animate_trajectories(self, df, filename="trajectories_animation.mp4"):
        # opcional: garantir backend não-interativo (útil em servidores)
        try:
            plt.switch_backend('Agg')
        except Exception:
            pass

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('Animated Trajectories')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.grid(True); ax.axis('equal')

        trajectories = {tid: traj.sort_values('timestamp') for tid, traj in df.groupby('trajectory_id')}
        if not trajectories:
            print("No trajectories to animate."); plt.close(fig); return

        max_len = max(len(traj) for traj in trajectories.values())

        lines = {}
        for traj_id in trajectories.keys():
            lines[traj_id], = ax.plot([], [], '-', label=f'Trajectory {traj_id}')

        ax.set_xlim(df['x'].min() - 1, df['x'].max() + 1)
        ax.set_ylim(df['y'].min() - 1, df['y'].max() + 1)
        ax.legend(title='Trajectory ID', loc='best', fontsize=8)

        def init():
            for line in lines.values():
                line.set_data([], [])
            return tuple(lines.values())

        def update(frame):
            for traj_id, traj in trajectories.items():
                if frame < len(traj):
                    data = traj.iloc[:frame+1]
                    lines[traj_id].set_data(data['x'], data['y'])
            return tuple(lines.values())

        # <-- mudanca importante -->
        ani = animation.FuncAnimation(fig, update, frames=max_len,
                                    init_func=init, blit=False, interval=50,
                                    repeat=False)

        animation_path = str(self.output_dir / filename)

        # checar ffmpeg
        if shutil.which("ffmpeg") is None:
            print("ffmpeg não encontrado no PATH. Execute 'ffmpeg -version' no terminal para confirmar.")
        try:
            print("Iniciando save para:", animation_path)
            writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='BehaviourRugs'), bitrate=1800)
            ani.save(animation_path, writer=writer)
            print(f"Animation saved to {animation_path}")
        except FileNotFoundError:
            print("Error: ffmpeg not found. Instale ffmpeg e coloque no PATH.")
        except Exception as e:
            print("Erro ao salvar animação:", repr(e))
        finally:
            plt.close(fig)
