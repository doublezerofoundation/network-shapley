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


def build_sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return small private_links, public_links, demand DataFrames."""
    private_links = pd.DataFrame(
        {
            "Start":     ["FRA1", "FRA1", "SIN1"],
            "End":       ["NYC1", "SIN1", "NYC1"],
            "Cost":      [40, 50, 80],
            "Bandwidth": [10, 10, 10],
            "Operator1": ['Alpha', 'Beta', 'Gamma'],
            "Operator2": [pd.NA, pd.NA, pd.NA],
            "Uptime":    [1, 1, 1],
            "Shared":    [pd.NA, pd.NA, pd.NA],
        }
    )

    public_links = pd.DataFrame(
        {
            "Start": ["FRA1", "FRA1", "SIN1"],
            "End":   ["NYC1", "SIN1", "NYC1"],
            "Cost":  [70, 80, 120],
        }
    )

    demand = pd.DataFrame(
        {
            "Start":   ["SIN", "SIN"],
            "End":     ["NYC", "FRA"],
            "Traffic": [5, 5],
            "Type":    [1, 1],
        }
    )

    return private_links, public_links, demand


def main() -> None:
    private_links, public_links, demand = build_sample_inputs()

    result = network_shapley(
        private_links=private_links,
        public_links=public_links,
        demand=demand,
        operator_uptime=0.98,
        hybrid_penalty=5.0,
        demand_multiplier=1.0,
    )

    print("\nShapley results:\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
