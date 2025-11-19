import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import pandas as pd
import numpy as np
from pathlib import Path
from datasketch import MinHash, MinHashLSH
from matplotlib.colors import hsv_to_rgb

class TrajectoryClusterer:
    """Clusters trajectories based on symbolic movement motifs."""

    def __init__(self, symbolic_df, k_motif=2, lsh_threshold=0.2, num_perm=128):
        self.symbolic_df = symbolic_df.copy()
        self.k = k_motif
        self.lsh_threshold = lsh_threshold
        self.num_perm = num_perm

    def _compute_mfv(self, sequence):
        """Computes the Motif Frequency Vector."""
        seq = [x for x in sequence if pd.notna(x) and x != ""]
        motifs = [tuple(seq[i:i+self.k]) for i in range(len(seq) - self.k + 1)]
        counts = Counter(motifs)
        total = sum(counts.values())
        return {motif: count / total for motif, count in counts.items()} if total > 0 else {}

    def _mfv_to_minhash(self, mfv):
        """Converts an MFV to a MinHash signature."""
        m = MinHash(num_perm=self.num_perm)
        for motif in mfv.keys():
            m.update(str(motif).encode('utf8'))
        return m

    def cluster(self):
        """Performs LSH clustering."""
        self.symbolic_df[f"MFV_k{self.k}"] = self.symbolic_df["simbolic_movement"].apply(self._compute_mfv)

        lsh = MinHashLSH(threshold=self.lsh_threshold, num_perm=self.num_perm)
        minhashes = {}

        for i, row in self.symbolic_df.iterrows():
            key = f"traj_{row['group']}"
            m = self._mfv_to_minhash(row[f"MFV_k{self.k}"])
            minhashes[key] = m
            lsh.insert(key, m)

        bucket_labels = {}
        for key in minhashes:
            similar = sorted(lsh.query(minhashes[key]))
            bucket_id = "_".join(similar)
            bucket_labels[key] = bucket_id
        
        self.symbolic_df["lsh_bucket"] = self.symbolic_df["group"].apply(lambda g: bucket_labels[f"traj_{g}"])
        
        unique_buckets = {b: idx for idx, b in enumerate(sorted(set(self.symbolic_df["lsh_bucket"])))}
        self.symbolic_df["cluster_id"] = self.symbolic_df["lsh_bucket"].map(unique_buckets)
        
        return self.symbolic_df.sort_values(by="cluster_id").reset_index(drop=True)
