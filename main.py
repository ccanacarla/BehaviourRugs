import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
import numpy as np

# Assuming trajectory_analysis modules are in the same directory or accessible
from trajectory_analysis.generatorDataset2 import TrajectoryGeneratorVariety
from trajectory_analysis import (
    TrajectoryClusterer,
    TrajectoryPreprocessor,
    FeatureExtractor,
    SymbolicConverter,
    Visualizer,
    VelocityAngleVisualizer,
)


class AnalysisPipeline:
    """
    Main class to orchestrate the trajectory analysis pipeline.
    """
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.get("output_dir", "."))
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def plot_trajectories(self, df, filename="ex1_traj_orig.png"):
        """Plots the generated trajectories."""
        plt.figure(figsize=(10, 10))

    # --- AGRUPAMENTO CORRETO ---
        grouped = df.groupby("trajectory_id")

        # Cria paleta de cores (uma cor por trajetória)
        colors = plt.cm.tab20(np.linspace(0, 1, len(grouped)))

        for color, (traj_id, group) in zip(colors, grouped):
            # Ordena por frame para garantir que a linha seja contínua
            group = group.sort_values("timestamp")

            plt.plot(
                group['x'],
                group['y'],
                linewidth=2,
                alpha=0.9,
                color=color,
                label=f"Traj {traj_id}"
            )

        # Estética do plot
        plt.title("Trajectories (grouped by trajectory_id)", fontsize=16)
        plt.xlabel("X", fontsize=12)
        plt.ylabel("Y", fontsize=12)


        plt.grid(True, linestyle="--", alpha=0.5)
        plt.axis("equal")

        # Legenda fora do gráfico
        plt.legend(
            title="Trajectory ID",
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            fontsize=8
        )

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_tsne_motifs(self, df_clustered, k_motif, filename="sintetico_tsne_motifs.png"):
        """
        Applies t-SNE on the Motif Frequency Vectors and plots the result.
        """
        print("\nStep 8: Applying t-SNE and plotting motif vectors...")
        
        mfv_column = f"MFV_k{k_motif}"
        if mfv_column not in df_clustered.columns:
            print(f"Motif Frequency Vector column '{mfv_column}' not found. Skipping t-SNE plot.")
            return

        # Ensure consistent order by sorting by cluster_id
        df_sorted = df_clustered.sort_values('cluster_id').reset_index(drop=True)

        motif_data = df_sorted[mfv_column].tolist()
        motif_df = pd.DataFrame(motif_data).fillna(0)

        if motif_df.empty or motif_df.shape[1] == 0:
            print("No motif data found to apply t-SNE. Skipping plot.")
            return

        perplexity_value = min(30, len(motif_df) - 1)
        if perplexity_value <= 0:
            print(f"Cannot run t-SNE with {len(motif_df)} samples. At least 2 are needed.")
            return
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
        tsne_results = tsne.fit_transform(motif_df)

        plt.figure(figsize=(12, 10))
        
        clusters = df_sorted['cluster_id']
        
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap='tab20')

        plt.title('t-SNE of Motif Frequency Vectors')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        # Create a legend with correct labels
        legend_labels = [f'Cluster {i}' for i in sorted(clusters.unique())]
        plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title='Cluster ID')
        
        plt.grid(True)
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved t-SNE plot to '{self.output_dir / filename}'")

    def run(self):
        """
        Executes the full analysis pipeline from data generation to visualization.
        """
        sufixo = "sintetico"
        # 0. Create dataset
        generator = TrajectoryGeneratorVariety(num_trajectories_per_type=5)
        #df_raw = generator.generate_dataset()
        #df_raw = pd.read_csv("128_fishing_trajsRugs10.csv")
        df_raw = pd.read_csv("visem47.csv")
        df_raw = df_raw.sort_values(["trajectory_id", "timestamp"])


        print("Step 1: Loading dataset...")
        
        self.plot_trajectories(df_raw, f"{sufixo}_traj_orig.png")
        print(f"Saved trajectory plot to '{self.output_dir}'")

        # 2. Preprocess Trajectories
        print("\nStep 2: Preprocessing trajectories (translate and rotate)...")
        preprocessor = TrajectoryPreprocessor(df_raw, output_dir=self.output_dir)
        df_processed = preprocessor.process()
        preprocessor.plot_comparison(f"{sufixo}_Comparacao_Original_Rotacionado_PCA.png")
        print(f"Saved preprocessed comparison plot to '{self.output_dir}'")

        # 3. Extract Features
        print("\nStep 3: Extracting features (speed and angle)...")
        feature_extractor = FeatureExtractor(df_processed, output_dir=self.output_dir)
        df_features = feature_extractor.extract_features()
        feature_extractor.plot_series("speed", "Temporal Series of Trajectory Speeds", "Speed", f"{sufixo}_Serie_vel.png")
        feature_extractor.plot_series("angle_deg", "Temporal Series of Trajectory Angular Variation", "Direction", f"{sufixo}_Serie_dir.png")
        df_features.to_csv(self.output_dir / f"{sufixo}_dataset_metrics.csv", index=False)
        print(f"Saved features dataset and plots to '{self.output_dir}'")

        # 3.5. Visualize Velocity-Angle Rugs
        print("\nStep 3.5: Generating Velocity-Angle Rugs visualization...")
        vel_angle_visualizer = VelocityAngleVisualizer(output_dir=self.output_dir)
        vel_angle_visualizer.plot_velocity_angle_rugs(df_features, f"{sufixo}_VelocityAngleRugs.png")
        vel_angle_visualizer.plot_legend(f"{sufixo}_VelocityAngleLegend.png")
        print(f"Saved Velocity-Angle Rugs and Legend plots to '{self.output_dir}'")

        # 4. Convert to Symbolic Representation
        print("\nStep 4: Converting features to symbolic representation...")
        symbolic_converter = SymbolicConverter(df_features)
        df_symbolic_lists = symbolic_converter.convert_to_symbols()
        print("Symbolic lists created.")

        # 5. Cluster Trajectories
        print("\nStep 5: Clustering trajectories using LSH...")
        clusterer = TrajectoryClusterer(k_motif=4, symbolic_df=df_symbolic_lists)
        df_clustered = clusterer.cluster()
        df_clustered.to_csv(self.output_dir / f"{sufixo}_dataset_simbolic.csv", index=False)
        print(f"Saved clustered symbolic dataset to '{self.output_dir}'")
        print("Clustering results:")
        print(df_clustered[['group', 'cluster_id', 'lsh_bucket']].head())

        # 6. Visualize Behaviour Rugs
        print("\nStep 6: Generating Behaviour Rugs visualization...")
        visualizer = Visualizer(output_dir=self.output_dir)
        visualizer.plot_behaviour_rugs(df_clustered, f"{sufixo}_BehaviourRugs.png")
        print(f"Saved Behaviour Rugs plot to '{self.output_dir}'")

        # 7. Visualize Motif Frequencies
        print("\nStep 7: Generating Motif Frequency visualization...")
        mfv_column = f"MFV_k{clusterer.k}"

        # Sort by cluster and then by group to ensure consistent order for plotting
        df_plot = df_clustered.sort_values(['cluster_id', 'group']).reset_index(drop=True)
        
        motif_data = df_plot[mfv_column].tolist()
        motif_df = pd.DataFrame(motif_data).fillna(0)

        if not motif_df.empty:
            # Create descriptive labels for the y-axis
            labels = "Traj: " + df_plot["group"].astype(str) + " (C:" + df_plot["cluster_id"].astype(str) + ")"
            motif_df.index = labels
            
            motif_df.columns = [str(c) for c in motif_df.columns]
            motif_df = motif_df.reindex(sorted(motif_df.columns), axis=1)

            visualizer.plot_motif_frequency(
                motif_df, 
                f"{sufixo}_MotifFrequency.png",
                ylabel="Trajectory ID (Cluster ID)"
            )
            print(f"Saved Motif Frequency plot to '{self.output_dir}'")
        else:
            print("No motifs found, skipping frequency plot.")

        # 8. Plot t-SNE of Motifs
        self.plot_tsne_motifs(df_clustered, clusterer.k, f"{sufixo}_tsne_motifs.png")

        print("Pipeline finished successfully!")

        # 9. Generate Trajectories Animation
        print("\nStep 9: Generating Trajectories Animation...")
        # Note: The df_processed should contain 'timestamp' column for the animation to work correctly
        if 'timestamp' not in df_processed.columns:
            print("Warning: 'timestamp' column not found in processed data. Animation may not be correct.")
            # Add a dummy timestamp if it's missing, assuming uniform time steps
            df_processed = df_processed.groupby('trajectory_id').apply(lambda g: g.assign(timestamp=range(len(g)))).reset_index(drop=True)

        visualizer.animate_trajectories(df_processed, f"{sufixo}_trajectories_animation.mp4")
        
        print("\nPipeline finished successfully!")


if __name__ == "__main__":
    # Configuration for the pipeline
    pipeline_config = {
        "output_dir": "output_analysis"  # all output will be saved in this directory
    }

    # Instantiate and run the main pipeline
    main_pipeline = AnalysisPipeline(pipeline_config)
    main_pipeline.run()
