"""
A minimal, self-contained demonstration of the network_shapley() function.

Run from the repo root with:
    python example_run.py
"""

from __future__ import annotations
import pathlib, sys
import pandas as pd

# --- Ensure the repo root is on PYTHONPATH so we can import network_shapley.py ---
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))          # allows `import network_shapley`

from network_shapley import network_shapley


def build_sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return small private_links, devices, public_links, demand DataFrames."""
    private_links = pd.DataFrame(
        {
            "Device1":   ["SIN1", "FRA1", "FRA1"],
            "Device2":   ["FRA1", "AMS1", "LON1"],
            "Latency":   [50, 3, 5],
            "Bandwidth": [10, 10, 10],
            "Uptime":    [1, 1, 1],
            "Shared":    [pd.NA, pd.NA, pd.NA],
        }
    )

    devices = pd.DataFrame(
        {
            "Device":   ["SIN1", "FRA1", "AMS1", "LON1"],
            "Edge":     [1, 1, 1, 1],
            "Operator": ["Alpha", "Alpha", "Beta", "Beta"],

        }
    )

    public_links = pd.DataFrame(
        {
            "City1":   ["SIN", "SIN", "FRA", "FRA"],
            "City2":   ["FRA", "AMS", "LON", "AMS"],
            "Latency": [100, 102, 7, 5],
        }
    )

    demand = pd.DataFrame(
        {
            "Start":     ["SIN", "SIN", "AMS", "AMS"],
            "End":       ["AMS", "LON", "LON", "FRA"],
            "Receivers": [1, 5, 2, 1],
            "Traffic":   [1, 1, 3, 3],
            "Priority":  [1, 2, 1, 1],
            "Type":      [1, 1, 2, 2],
            "Multicast": [True, True, False, False]
        }
    )

    return private_links, devices, public_links, demand


def main() -> None:
    private_links, devices, public_links, demand = build_sample_inputs()

    result = network_shapley(
        private_links=private_links,
        devices=devices,
        demand=demand,
        public_links=public_links,
        operator_uptime=0.98,
        contiguity_bonus=5.0,
        demand_multiplier=1.0,
    )

    print("\nShapley results:\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
