import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class SymbolicConverter:
    """Discretizes features into symbolic representations."""

    def __init__(self, df):
        self.df = df.copy()

    def _label_speed(self, v, eps_stop, q25, q75):
        if np.isnan(v): return np.nan
        if v <= eps_stop: return "Parado"
        if v <= q25: return "Lento"
        if v <= q75: return "Médio"
        return "Rápido"

    def _angle_to_dir8(self, a_deg):
        if np.isnan(a_deg): return np.nan
        a = ((a_deg + 180) % 360) - 180
        if -22.5 < a <= 22.5: return "Leste"
        if 22.5 < a <= 67.5: return "Nordeste"
        if 67.5 < a <= 112.5: return "Norte"
        if 112.5 < a <= 157.5: return "Noroeste"
        if a > 157.5 or a <= -157.5: return "Oeste"
        if -157.5 < a <= -112.5: return "Sudoeste"
        if -112.5 < a <= -67.5: return "Sul"
        return "Sudeste"

    def convert_to_symbols(self):
        """Converts continuous features to discrete symbols."""
        # Speed discretization
        median_speed = float(np.nanmedian(self.df["speed"]))
        eps_stop = 1e-6 if median_speed == 0 else 1e-3 * median_speed
        moving_mask = self.df["speed"] > eps_stop
        q25, q75 = np.quantile(self.df.loc[moving_mask, "speed"], [0.25, 0.75]) if moving_mask.any() else (0, 0)
        
        self.df["speed_label"] = self.df["speed"].apply(self._label_speed, args=(eps_stop, q25, q75))
        
        # Direction discretization
        self.df["dir8_label"] = self.df["angle_deg"].apply(self._angle_to_dir8)
        
        # Combine into a single symbol
        def make_lists(g):
            g = g.sort_values("timestamp")
            return pd.Series({
                "speed_list": g["speed_label"].tolist(),
                "direction_list": g["dir8_label"].tolist()
            })

        lists_df = self.df.groupby("trajectory_id", as_index=False).apply(make_lists)
        lists_df = lists_df.rename(columns={"trajectory_id": "group"})

        lists_df["simbolic_movement"] = lists_df.apply(
            lambda row: [
                str(s) + str(d) if not (pd.isna(s) or pd.isna(d)) else ""
                for s, d in zip(row["speed_list"], row["direction_list"])
            ],
            axis=1
        )
        return lists_df