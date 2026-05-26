"""
fix_data.py
===========
Cleans the raw VicRoads SCATS export (data/scats_clean.csv) so it can be
consumed by data_processing.py and routing.py.

Steps
-----
1. Read the CSV with no assumed header (header=None).
2. Drop row 0 if it contains garbage text like "Start Time".
3. Assign the canonical 106-column schema (10 metadata + 96 volume columns).
4. Convert the "Date" column from Excel serial numbers (e.g. 39020.01) to
   the standard 'dd/mm/yyyy' string format used by the rest of the pipeline.
5. Save the result back to data/scats_clean.csv (comma-separated, no index).

Usage
-----
    python fix_data.py
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "data", "scats_clean.csv")

# ── Canonical column schema ──────────────────────────────────────────────
META_COLS   = [
    "SCATS Number", "Location", "Site_Type",
    "NB_LATITUDE", "NB_LONGITUDE",
    "ALL_VEHS", "ALL_TRKS", "TRK_PCT", "Unknown", "Date",
]
VOLUME_COLS = [f"V{i:02d}" for i in range(96)]
ALL_COLS    = META_COLS + VOLUME_COLS          # 106 columns total

# Excel serial-date epoch (accounts for Excel's 1900 leap-year bug)
_EXCEL_BASE = pd.Timestamp("1899-12-30")

# Substrings that identify a garbage header row
_GARBAGE_KEYWORDS = {"start time", "scats", "location", "date", "v00"}


def _excel_serial_to_datestr(value: object) -> str:
    """
    Convert an Excel serial number (int or float) to 'dd/mm/yyyy'.
    If the value already looks like a date string, return it unchanged.
    """
    s = str(value).strip()
    if "/" in s or "-" in s:
        return s                           # already a date string
    try:
        ts = _EXCEL_BASE + pd.Timedelta(days=float(s))
        return ts.strftime("%d/%m/%Y")
    except (ValueError, OverflowError):
        return s                           # pass through anything unrecognised


def _is_garbage_row(row: pd.Series) -> bool:
    """Return True if the row looks like a non-data header or label row."""
    tokens = " ".join(str(v).lower() for v in row.values)
    return any(kw in tokens for kw in _GARBAGE_KEYWORDS)


def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Input file not found: {CSV_PATH}\n"
            "Export scats_clean.csv from VicRoads Excel and place it in data/."
        )

    print(f"[fix_data] Reading  : {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, header=None, dtype=str, low_memory=False)
    print(f"           Raw shape: {df.shape}")

    # ── Step 1: Drop garbage header row ─────────────────────────────────
    if _is_garbage_row(df.iloc[0]):
        print("           Dropping garbage header row (row 0).")
        df = df.iloc[1:].reset_index(drop=True)

    # ── Step 2: Assign canonical column names ────────────────────────────
    n_cols = df.shape[1]
    if n_cols != len(ALL_COLS):
        raise ValueError(
            f"Column count mismatch: expected {len(ALL_COLS)}, found {n_cols}.\n"
            "Check that the export matches the standard SCATS 106-column layout."
        )
    df.columns = ALL_COLS
    print(f"           Assigned {len(ALL_COLS)} column names.")

    # ── Step 3: Convert Excel serial dates ───────────────────────────────
    sample_raw = str(df["Date"].iloc[0]).strip()
    if "/" in sample_raw or "-" in sample_raw:
        print(f"           Date column already string format ({sample_raw!r}). Skipping.")
    else:
        print(f"           Converting Excel serial dates (sample: {sample_raw!r}) …")
        df["Date"] = df["Date"].apply(_excel_serial_to_datestr)
        print(f"           Sample converted: {df['Date'].iloc[0]!r}")

    # ── Step 4: Tidy string columns ──────────────────────────────────────
    for col in ["SCATS Number", "Location", "NB_LATITUDE", "NB_LONGITUDE"]:
        df[col] = df[col].astype(str).str.strip()

    # ── Step 5: Save ─────────────────────────────────────────────────────
    df.to_csv(CSV_PATH, index=False)
    print(f"           Saved    : {CSV_PATH}  ({len(df):,} rows × {df.shape[1]} cols)")
    print("[fix_data] Done.")


if __name__ == "__main__":
    main()