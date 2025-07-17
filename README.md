# network-shapley Guide

A Python script to compute Shapley values for network contributors. Note that this is the repo for prototyping and analysis. The production version of this work will be in Rust: [network-shapley-rs](https://github.com/doublezerofoundation/network-shapley-rs).

## Overview

`network_shapley.py` lets you combine a list of private links and the associated devices, a list of public links, and a matrix of demand to get Shapley values, i.e. the marginal contribution to overall network performance.

The result is a table giving every operator’s absolute value created and their share of the total reward.

## Background

Shapley values are a tool in cooperative game theory to align incentives, such that a contributor is paid the true marginal value of a contribution. This contrasts with the traditional carried-traffic model, which simply credits links in proportion to the number of gigabits they move. The carried-traffic model looks intuitive, but it ignores complications around redundancy, heterogeneous performance, benchmarks, etc. The Shapley value model, by contrast, evaluates every possible coalition of contributors and stacks improvements to a universal value function over a baseline world with only the public internet. In this approach, the contributor earns exactly the share of rewards that corresponds to the incremental value it unlocks. This makes it impossible to free‑ride on already‑congested routes or to over‑charge for vanity capacity that fails to improve end‑to‑end latency.

## Get Started

To download and demonstrate the script:

```bash
git clone https://github.com/doublezerofoundation/network-shapley.git
cd network-shapley
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python example_run.py
```

## Build Your Own Worlds

You can build more complex simulations and estimate the reward split accordingly. Below is the usage needed to try two model simulations packaged into the repo, for a given setup of private links, devices, and public links, and two different demand profiles. In these simulations, eight network contributors contribute a total of twelve simulated links that connect twelves distinct devices across nine cities. In the first demand profile, a leader located in Singapore sends out blocks approximately commensurate with the Solana stake to the other eight cities, while an RPC server in LAX also streams data at that leader. In the second demand profile, a leader located in New York sends out blocks approximately commensurate with the Solana stake to the other eight cities, while a Buenos Aires-based leader in second new blockchain sends equal traffic to all other cities.

```python
import pandas as pd
from network_shapley import network_shapley

private_links = pd.read_csv("private_links.csv")
devices       = pd.read_csv("devices.csv")
public_links  = pd.read_csv("public_links.csv")
demand1       = pd.read_csv("demand1.csv")
demand2       = pd.read_csv("demand2.csv")

result1 = network_shapley(
    private_links     = private_links,
    devices           = devices,
    demand            = demand1,
    public_links      = public_links,
    operator_uptime   = 0.98, # optional
    contiguity_bonus  = 5.0,  # optional
    demand_multiplier = 1.2,  # optional
)
print(result1)

result2 = network_shapley(
    private_links     = private_links,
    devices           = devices,
    demand            = demand2,
    public_links      = public_links,
    operator_uptime   = 0.98, # optional
    contiguity_bonus  = 5.0,  # optional
    demand_multiplier = 1.2,  # optional
)
print(result2)
```

It is important to stress that the links, latencies, and other components of these example files are entirely invented. They are meant to illustrate the methodology and inputs, but users should construct their own simulations accordingly.

## Inputs and Usage

| Argument            | Schema                                                                                                             | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `private_links`     | `pandas.DataFrame` with columns: `Device1`, `Device2`, `Latency`, `Bandwidth`, `Uptime`, `Shared`                  | Must contain exactly one row for each physical private link. For each row, indicate first device endpoints, the latency, and the bandwidth available. Next, `Uptime` indicates the link reliability, e.g. `0.99`. Finally, `Shared` indicates whether the link shares bandwidth with another: set as `NA` for links that have exclusive bandwidth, and set to the same number for all rows that do share bandwidth. List each physical link only once; the code automatically mirrors it for the reverse direction.                                                                                                                                                                                                                                                                                                                                               |
| `devices`           |  `pandas.DataFrame` with columns: `Device`, `Edge`, and `Operator`                                                 | Must contain one row for each device in `private_links`. For each row, indicate the device name, the edge (i.e. internet access) connection it supports, and the operator name. Note that device labels should be a city followed by digits (e.g. `AMS1` or `FRA01`). Finally, do not use `Public` for the name of `Operator`; this is a reserved keyword for public links only.                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `demand`            | `pandas.DataFrame` with columns: `Start`, `End`, `Receivers`, `Traffic`, `Priority`, `Type`, `Multicast`           | Must contain exactly one row for each traffic flow between a sender and recipient node. For each row, indicate the starting and ending nodes; nodes here are cities and must not contain digits (e.g. `FRA`). Next, indicate the number of receivers for that traffic flow, the amount of traffic each receiver gets, and the amount of priority (e.g. stake weight) each receiver in that table has. Next, use `Type` to distinguish between different flows: every row of the same type should have the same sender, same traffic, and same multicast flag (next). Finally, indicate whether the flow can utilize multicast by marking a boolean for `Multicast`.                                                                                                                                                                                              |
| `public_links`      | `pandas.DataFrame` with columns: `City1`, `City2`, `Latency`                                                       | Must contain one row for each pair of cities that have a measured public latency between them. For each row, indicate city endpoints and the latency; again note that cities should not contain digits (e.g. `FRA`). The graph does not have to be perfectly connected, but all cities where demand starts or ends in `demand` must have at least one public pathway measured.  List each combination only once; the code automatically mirrors it for the reverse direction.                                                                                                                                                                                                                                                                                                                                                                                   |
| `operator_uptime`   | optional `float`: default `1.0`                                                                                    | Probability a given operator is available in an epoch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `contiguity_bonus`  | optional `float`: default `5.0`                                                                                    | Extra latency penalty for usage of hybrid paths that mix private and public routes, i.e. penalty that is avoided for using contiguous networks.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `demand_multiplier` | optional `float`: default `1.0`                                                                                    | Scales traffic demand, to future-proof network.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

Users need only to call `network_shapley` with these inputs. There are other helper functions, e.g. `consolidate_links` and `lp_primitives`, but they are not to be called directly by the end user except for testing purposes.

The output is a single `pandas.DataFrame` with three columns: `Operator`, `Value`, and `Percent`, which respectively note the operator, the raw Shapley value for that operator, and the percentage it represents of all Shapley values.

`network_shapley` is fully vectorised with NumPy/SciPy and remains quick for networks of a few hundred links and a dozen operators.

## Limits and Troubleshooting

The script currently limits to 15-20 operators to keep the computations manageable. Feel free to remove the limit. The script does not impose a limit on the number of links.

The script does some light error checking but it does not yet check for all corner cases. Please open an issue or contact nihar@doublezero.us if you are unable to get a particular simulation working.

Please read the manual or watch the video walkthrough at https://youtu.be/K1Ni-k51sMw for a richer understanding of the model.
