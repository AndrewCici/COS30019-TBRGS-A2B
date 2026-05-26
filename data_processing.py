import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


class TrafficDataProcessor:
    """
    Processes SCATS traffic volume CSV data into PyTorch DataLoader objects
    suitable for RNN-based time-series forecasting.
    """

    def __init__(self, filepath: str, time_step: int = 12, batch_size: int = 32):
        self.filepath = filepath
        self.time_step = time_step
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.volume_columns = [f"V{i:02d}" for i in range(96)]

    # ------------------------------------------------------------------
    # 1. Load & clean
    # ------------------------------------------------------------------
    def _load_data(self) -> pd.DataFrame:
        """
        Load the SCATS CSV (semicolon-delimited, two header rows).
        The first row contains human-readable time labels; the second row
        contains the machine-readable column names used throughout this class.
        """
        df = pd.read_csv(self.filepath, sep=";", header=1)
        return df

    def _handle_missing(self, series: pd.Series) -> pd.Series:
        """
        Fill missing values: linear interpolation first, then forward-fill
        for any NaNs that remain at the boundaries.
        """
        series = series.interpolate(method="linear", limit_direction="both")
        series = series.ffill().bfill()
        return series

    # ------------------------------------------------------------------
    # 2. Extract time-series volume data
    # ------------------------------------------------------------------
    def _extract_volume_series(self, df: pd.DataFrame) -> np.ndarray:
        """
        Melt wide-format volume columns (V00–V95) into a single 1-D
        chronological time-series ordered by (Date, interval index).

        Returns
        -------
        np.ndarray of shape (n_timesteps, 1)
        """
        # Keep only volume columns and the date so we can sort correctly
        volume_df = df[["Date"] + self.volume_columns].copy()

        # Parse dates (format: D/M/YY)
        volume_df["Date"] = pd.to_datetime(volume_df["Date"], dayfirst=True)
        volume_df = volume_df.sort_values("Date").reset_index(drop=True)

        # Melt: one row per (date, 15-min interval)
        melted = volume_df.melt(
            id_vars="Date", value_vars=self.volume_columns,
            var_name="interval", value_name="volume"
        )

        # Preserve temporal order: sort by date then by interval index (V00 < V01 …)
        melted["interval_idx"] = melted["interval"].str[1:].astype(int)
        melted = melted.sort_values(["Date", "interval_idx"]).reset_index(drop=True)

        # Convert to numeric and handle missing values
        melted["volume"] = pd.to_numeric(melted["volume"], errors="coerce")
        melted["volume"] = self._handle_missing(melted["volume"])

        return melted["volume"].values.reshape(-1, 1)

    # ------------------------------------------------------------------
    # 3. Scale
    # ------------------------------------------------------------------
    def _scale(self, data: np.ndarray) -> np.ndarray:
        """Fit-transform on the full series; inverse_transform is available
        via self.scaler after calling build_loaders()."""
        return self.scaler.fit_transform(data)

    # ------------------------------------------------------------------
    # 4. Sliding-window sequence generator
    # ------------------------------------------------------------------
    def _create_sequences(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build supervised (X, y) pairs using a sliding window of size
        `self.time_step`.

        Parameters
        ----------
        data : np.ndarray, shape (n, 1)

        Returns
        -------
        X : np.ndarray, shape (n_samples, time_step, 1)   ← (batch, seq, features)
        y : np.ndarray, shape (n_samples, 1)
        """
        X, y = [], []
        for i in range(len(data) - self.time_step):
            X.append(data[i : i + self.time_step])          # (time_step, 1)
            y.append(data[i + self.time_step])               # (1,)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ------------------------------------------------------------------
    # 5. Train / Validation / Test split (no shuffling)
    # ------------------------------------------------------------------
    @staticmethod
    def _split(
        X: np.ndarray, y: np.ndarray, train_ratio: float = 0.70, val_ratio: float = 0.15
    ) -> tuple:
        n = len(X)
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))

        X_train, y_train = X[:train_end],        y[:train_end]
        X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
        X_test,  y_test  = X[val_end:],          y[val_end:]

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    # ------------------------------------------------------------------
    # 6. Build DataLoaders
    # ------------------------------------------------------------------
    @staticmethod
    def _make_loader(
        X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
    ) -> DataLoader:
        """
        Convert numpy arrays to a PyTorch DataLoader.

        Input tensor shape: (batch, sequence, features) — required by
        PyTorch nn.RNN / nn.LSTM / nn.GRU when batch_first=True.
        """
        X_t = torch.tensor(X)   # (n, time_step, 1) — already float32
        y_t = torch.tensor(y)   # (n, 1)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_loaders(
        self,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Full pipeline: load → clean → scale → sequence → split → DataLoaders.

        Returns
        -------
        train_loader, val_loader, test_loader : torch.utils.data.DataLoader
            Each batch yields (X, y) where:
              X : Tensor of shape (batch_size, time_step, 1)
              y : Tensor of shape (batch_size, 1)
        """
        # Step 1 – Load
        df = self._load_data()

        # Step 2 – Extract & handle missing
        volume_series = self._extract_volume_series(df)   # (N, 1)

        # Step 3 – Scale
        scaled = self._scale(volume_series)               # (N, 1), values in [0, 1]

        # Step 4 – Sliding window
        X, y = self._create_sequences(scaled)             # (M, time_step, 1), (M, 1)

        # Step 5 – Temporal split
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self._split(X, y)

        print(
            f"[TrafficDataProcessor] Series length : {len(volume_series):,}\n"
            f"  Total sequences : {len(X):,}\n"
            f"  Train           : {len(X_train):,}\n"
            f"  Validation      : {len(X_val):,}\n"
            f"  Test            : {len(X_test):,}\n"
            f"  Input shape     : {X_train.shape}  → (samples, time_step={self.time_step}, features=1)"
        )

        # Step 6 – DataLoaders (only train is shuffled)
        train_loader = self._make_loader(X_train, y_train, self.batch_size, shuffle=False)
        val_loader   = self._make_loader(X_val,   y_val,   self.batch_size, shuffle=False)
        test_loader  = self._make_loader(X_test,  y_test,  self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "scats_sample.csv")

    processor = TrafficDataProcessor(
        filepath=DATA_PATH,
        time_step=12,
        batch_size=32,
    )
    train_loader, val_loader, test_loader = processor.build_loaders()

    # Verify shapes
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nSample batch — X: {tuple(X_batch.shape)}, y: {tuple(y_batch.shape)}")
    assert X_batch.ndim == 3, "Expected 3-D input tensor (batch, seq, features)"
    assert X_batch.shape[1] == 12, "Sequence length must equal time_step=12"
    assert X_batch.shape[2] == 1,  "Feature dimension must be 1"
    print("All shape assertions passed.")
