import pandas as pd
import matplotlib.pyplot as plt   # ✅ Corrigido: era "import matplotlib as plt"
from pathlib import Path          # ✅ Necessário para manipular diretórios de saída
from trajectory_analysis import (
    TrajectoryClusterer,
    TrajectoryPreprocessor,
    FeatureExtractor,
    SymbolicConverter,
    Visualizer
)


class AnalysisPipeline:
    """
    Main class to orchestrate the trajectory analysis pipeline.
    """
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.get("output_dir", "."))
        self.output_dir.mkdir(exist_ok=True, parents=True)  # ✅ garante que o diretório exista

    def plot_trajectories(self, df, filename="ex1_traj_orig.png"):
        """Plots the generated trajectories."""
        plt.figure(figsize=(10, 10))   # ✅ corrigido: 1000x1000 era exagerado
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

    def run(self):
        """
        Executes the full analysis pipeline from data generation to visualization.
        """
        sufixo = "sintetico"

        # 1. Load dataset and rename columns
        print("Step 1: Loading dataset...")
        #df_raw = pd.read_csv("user47_long_trajectories.csv")

        #df_raw = pd.read_csv("ex1_synthetic_trajectories.csv")
        df_raw = pd.read_csv("128_fishing_trajsRugs10.csv")
        #ids_10 = df_raw["trajectory_id"].unique()[:20]

        # --- Filtrar o dataframe ---
        #df_raw = df_raw[df_raw["trajectory_id"].isin(ids_10)].copy()

        # ✅ Plot trajectories
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

        # 4. Convert to Symbolic Representation
        print("\nStep 4: Converting features to symbolic representation...")
        symbolic_converter = SymbolicConverter(df_features)
        df_symbolic_lists = symbolic_converter.convert_to_symbols()
        print("Symbolic lists created.")

        # 5. Cluster Trajectories
        print("\nStep 5: Clustering trajectories using LSH...")
        clusterer = TrajectoryClusterer(k_motif=3, symbolic_df=df_symbolic_lists)
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

        print("Pipeline finished successfully!")

         # 7. Visualize Motif Frequencies
        print("\nStep 7: Generating Motif Frequency visualization...")
        mfv_column = f"MFV_k{clusterer.k}"
        motif_data = df_clustered[mfv_column].tolist()
        motif_df = pd.DataFrame(motif_data).fillna(0)
        if not motif_df.empty:
            motif_df.columns = [str(c) for c in motif_df.columns]
            motif_df = motif_df.reindex(sorted(motif_df.columns), axis=1)
            visualizer.plot_motif_frequency(motif_df, f"{sufixo}_MotifFrequency.png")
            print(f"Saved Motif Frequency plot to '{self.output_dir}'")
        else:
            print("No motifs found, skipping frequency plot.")

        # 8. Generate Trajectories Animation
        print("\nStep 8: Generating Trajectories Animation...")
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
