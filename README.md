# network-shapley Guide
A Python script to compute Shapley values for network contributors

## Overview
```network_shapley.py``` lets you combine a map of private links, a map of public links, and a matrix of demand to get Shapley values, i.e. the marginal contribution to overall network performance.

The result is a table giving every operator’s absolute value created and their share of the total reward.

## Get Started
To download and demonstrate the script:
```bash
git clone https://github.com/nihar-dz/network-shapley.git
cd network-shapley
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python example_run.py
```

## Build Your Own Worlds
You can build more complex simulations and estimate the reward split accordingly. Below is the usage needed to try two model simulations packaged into the repo, for a given set of private links and public links, and two different demand profiles. In these simulations, eight network contributors contribute a total of twelve simulated links that connect twelves distinct switches across nine cities. In the first demand profile, a leader located in Singapore sends out blocks approximately commensurate with the Solana stake to the other eight cities, while an RPC server in LAX also streams data at that leader. In the second demand profile, a leader located in New York sends out blocks approximately commensurate with the Solana stake to the other eight cities, while a Buenos Aires-based leader in second new blockchain sends equal traffic to all other cities.

```python
import pandas as pd
from network_shapley import network_shapley

private_links = pd.read_csv("private_links.csv")
public_links  = pd.read_csv("public_links.csv")
demand1       = pd.read_csv("demand1.csv")
demand2       = pd.read_csv("demand2.csv")

result1 = network_shapley(
    private_links     = private_links,
    demand            = demand1,
    public_links      = public_links,
    operator_uptime   = 0.98, # optional
    hybrid_penalty    = 5.0,  # optional
    demand_multiplier = 1.2,  # optional
)
print(result1)

result2 = network_shapley(
    private_links     = private_links,
    demand            = demand2,
    public_links      = public_links,
    operator_uptime   = 0.98, # optional
    hybrid_penalty    = 5.0,  # optional
    demand_multiplier = 1.2,  # optional
)
print(result2)
```

It is important to stress that the links, latencies, and other components of these example files are entirely invented. They are meant to illustrate the methodology and inputs, but users should construct their own simulations accordingly.

## Inputs

| Argument | Schema | Notes |
| ----------- | ----------- | ----------- |
| ```private_links``` | ```pandas.DataFrame``` with columns: ```Start```, ```End```, ```Cost```, ```Bandwidth```, ```Operator1```, ```Operator2```, ```Uptime```, ```Shared``` | Must contain exactly one row for each physical private link. For each row, indicate switch endpoints, the latency (i.e. cost), the bandwidth available, and the primary operator in ```Operator1```. The column ```Operator2``` allows for dual ownership of a link but this is not currently used in the analysis, so it should be set as ```NA```. ```Uptime``` indicates the link reliability, e.g. ```0.99```. Finally, ```Shared``` indicates whether the link shares bandwidth with another: set as ```NA``` for links that have exclusive bandwidth, and set to the same number for all rows that do share bandwidth. List each physical link only once; the code automatically mirrors it for the reverse direction. Note that switch labels must contain a digit (e.g. ```FRA1```).  Finally, do not use ```0``` for the name of ```Operator1```; this is a reserved keyword for public links only. |
| ```public_links``` | ```pandas.DataFrame``` with columns: ```Start```, ```End```, ```Cost``` | Must contain exactly one row for each pair of switches that has a private path and/or demand endpoints, correpsonding to the public latency between them. For each row, indicate switch endpoints and the latency (i.e. cost). List each combination only once; the code automatically mirrors it for the reverse direction. Note again that switch labels must contain a digit (e.g. ```FRA1```). |
| ```demand``` | ```pandas.DataFrame``` with columns: ```Start```, ```End```, ```Traffic```, ```Type``` | Must contain exactly one row for each sender and recipient node, along with the quantity of traffic and a tag for the type of traffic. Nodes are cities and must not contain digits (e.g. ```FRA```). Every commodity ```Type``` must originate from a single ```Start``` node. |
| ```operator_uptime``` | optional ```float```: default ```1.0``` | Probability a given operator is available in an epoch. |
| ```hybrid_penalty``` | optional ```float```: default ```5.0``` | Extra latency penalty for usage of hybrid paths that mix private and public routes. |
| ```demand_multiplier``` | optional ```float```: default ```1.0``` | Scales traffic demand, to future-proof network. |

Users need only to call ```network_shapley``` with these inputs. There are other helper functions, e.g. ```consolidate_map``` and ```lp_primitives```, but they are not to be called directly by the end user except for testing purposes.

```network_shapley``` is fully vectorised with NumPy/SciPy and remains quick for networks of a few hundred links and up to fifteen operators.

## Limits and Troubleshooting
The script currently limits to fifteen operators to keep the computations manageable. Feel free to remove the limit. The script does not impose a limit on the number of links.

The script does some light error checking but it does not yet check for all corner cases. Please open an issue or contact nihar@doublezero.us if you are unable to get a particular simulation working.
