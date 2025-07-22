# Packages
from __future__ import annotations
import math
from typing import List, Dict, Sequence, Any
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.sparse import (
    csr_matrix,
    block_diag,
    diags,
    hstack as sp_hstack,
    vstack as sp_vstack,
)

# Helper utilities
def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)

def _has_digit(s: pd.Series) -> pd.Series:
    return s.str.contains(r"[0-9]")

def _unique_int(s: pd.Series) -> pd.Series:
    return s.map({u: i + 1 for i, u in enumerate(pd.unique(s))})

def _rep(arr: NDArray, times: int) -> NDArray:
    return np.tile(arr, times)

def _bits(n_bits: int) -> NDArray:
    # return (n_bits × 2^n_bits) bitmap where column j is binary of j
    cols = np.arange(2**n_bits, dtype=np.uint32)
    return ((cols[None] >> np.arange(n_bits)[:, None]) & 1).astype(np.uint8)

def _fact(v: NDArray) -> NDArray:
    return np.vectorize(math.factorial, otypes=[float])(v)

def check_inputs(
    private_links: pd.DataFrame,
    devices:       pd.DataFrame,
    demand:        pd.DataFrame,
    public_links:  pd.DataFrame,
    operator_uptime: float,
) -> None:
    """
    Checks for integrity in inputs and raises an error when a condition fails.

    Parameters
    ----------
    private_links : pandas.DataFrame
        Private link table `[Device1, Device2, Latency, Bandwidth, Uptime, Shared]`
    devices : pandas.DataFrame
        Device table `[Device, Edge, Operator]`
    demand : pandas.DataFrame
        Demand matrix `[Start, End, Receivers, Traffic, Priority, Type, Multicast, Original]`
    public_links : pandas.DataFrame
        Public internet links `[City1, City2, Latency]`
    operator_uptime : float
        Reliability (between 0 - 1) of an operator in any given epoch
    """

    # Check the operator count
    operators = np.sort([x for x in pd.unique(devices["Operator"].dropna().astype(str)) if x != 'Private'])
    _assert("Public" not in operators, "Public is a protected keyword for operator names; choose another.")
    n_ops = len(operators)
    if operator_uptime <= 0.99999999:
        _assert(n_ops < 16, "Too many operators; we limit to 15 to prevent the program from crashing.")
    _assert(n_ops < 21,
            "Too many operators; we limit to 20 when operator_uptime = 1 to prevent the program from crashing.")

    # Check that private links table is labeled correctly
    _assert(private_links.shape[0] > 0, "There must be at least one private link for this simulation.")
    _assert(_has_digit(private_links["Device1"]).all() & _has_digit(private_links["Device2"]).all(),
            "Devices are not labeled correctly in private links; they should be denoted with an integer.")
    _assert(~private_links["Device1"].astype(str).str.contains("00").any() &
            ~private_links["Device2"].astype(str).str.contains("00").any(),
            "Devices are not labeled correctly in the private_links table; they should not have a 00 code.")

    # Check that public links table is labeled correctly
    _assert(~_has_digit(public_links["City1"]).any() & ~_has_digit(public_links["City2"]).any(),
            "Cities are not labeled correctly in public links; they should not be denoted with an integer.")

    # Check that demand points are labeled correctly
    _assert(~_has_digit(demand["Start"]).any() & ~_has_digit(demand["End"]).any(),
            "Cities are not labeled correctly in public links; they should not be denoted with an integer.")

    # Check that, for a given demand type, there is a single origin, size, and multicast flag
    grp = demand.groupby("Type")
    _assert((grp["Start"].nunique() == 1).all() & (grp["Traffic"].nunique() == 1).all() &
            (grp["Multicast"].nunique() == 1).all(),"Demand types are not represented correctly.")

    # Check there are no duplicates devices
    _assert(~devices["Device"].duplicated().any(),"There are duplicated devices in the list.")

    # Check that every switch in private_links appears in devices
    _assert((set(private_links["Device1"]) | set(private_links["Device2"])).issubset(set(devices["Device"])),
            "Not all devices are in the device table.")

    # Check that all demand nodes are reachable by the public network
    public_nodes = set(public_links["City1"]) | set(public_links["City2"])
    _assert((set(demand["Start"]) | set(demand["End"])).issubset(public_nodes),
            "Demand is not fully linked to the public internet.")

def consolidate_demand(
    demand: pd.DataFrame,
    demand_multiplier: float,
) -> pd.DataFrame:
    """
    Construct a validated and augmented demand table, for lp_primitives()

    Parameters
    ----------
    demand : pandas.DataFrame
        Demand matrix `[Start, End, Receivers, Traffic, Priority, Type, Multicast]`
    demand_multiplier: float
        Extra multiplier to scale up demand

    Returns pandas.DataFrame
        A validated and augmented demand table ready for construction of the linear program
    """

    # Work on a copy to avoid mutating caller data
    demand_df = demand.copy()

    # Roll up cases with identical types, priorities, and destinations
    demand_df["priority_r2"] = demand_df["Priority"].round(2)
    duplicate_groups = demand_df.groupby(["Type", "End", "priority_r2"], as_index = False).size().query("size > 1")
    if not duplicate_groups.empty:
        rows_to_drop, aggregated = [], []
        for _, g in duplicate_groups.iterrows():
            sub = demand_df.loc[(demand_df["Type"] == g["Type"]) & (demand_df["End"] == g["End"]) &
                                (demand_df["priority_r2"] == g["priority_r2"])]
            aggregated.append({"Start": sub["Start"].iloc[0], "End": sub["End"].iloc[0],
                               "Receivers": sub["Receivers"].sum(), "Traffic": sub["Traffic"].iloc[0],
                               "Priority": sub["Priority"].mean(), "Type": sub["Type"].iloc[0],
                               "Multicast": sub["Multicast"].iloc[0]})
            rows_to_drop.extend(sub.index)
        demand_df = pd.concat([demand_df.drop(index=rows_to_drop), pd.DataFrame(aggregated)], ignore_index=True)
    demand_df = demand_df.drop(columns="priority_r2")

    # Retain original type before adjusting demand
    demand_df["Original"] = demand_df["Type"]

    # For unicast, split into unique types by rounded priority
    unicast_flag = ~demand_df["Multicast"]
    for t, grp in demand_df[unicast_flag].groupby("Type"):
        keys = grp["Priority"].round(2).unique()
        if len(keys) > 1:
            mapping = {k: t if i == 0 else demand_df["Type"].max() + i for i, k in enumerate(sorted(keys))}
            demand_df.loc[unicast_flag & (demand_df["Type"] == t), "Type"] = (
                demand_df.loc[unicast_flag & (demand_df["Type"] == t), "Priority"].round(2).map(mapping)
            )

    # For multicast, split into unique types for each row
    multicast_flag = demand_df["Multicast"]
    keys = demand_df[multicast_flag].groupby("Type", as_index=False).size().query("size > 1")
    if not keys.empty:
        for _, row in keys.iterrows():
            idx = demand_df.index[multicast_flag & (demand_df["Type"] == int(row["Type"]))]
            demand_df.loc[idx, "Type"] = ([int(row["Type"])] + list(range(demand_df["Type"].max() + 1,
                                                                          demand_df["Type"].max() + int(row["size"]))))

    # Multiply traffic by scaling factor
    demand_df['Traffic'] *= demand_multiplier

    return demand_df

def consolidate_links(
    private_links: pd.DataFrame,
    devices : pd.DataFrame,
    demand: pd.DataFrame,
    public_links: pd.DataFrame,
    contiguity_bonus: float,
) -> pd.DataFrame:
    """
    Construct a single and fully-validated link table, using both private and public links, for lp_primitives()

    Parameters
    ----------
    private_links : pandas.DataFrame
        Private link table `[Device1, Device2, Latency, Bandwidth, Uptime, Shared]`
    devices : pandas.DataFrame
        Device table `[Device, Edge, Operator]`
    demand : pandas.DataFrame
        Demand matrix `[Start, End, Receivers, Traffic, Priority, Type, Multicast, Original]`
    public_links : pandas.DataFrame
        Public internet links `[City1, City2, Latency]`
    contiguity_bonus : float
        Extra latency effectively added for mixing public links with private links

    Returns pandas.DataFrame
        A fully‑expanded, bidirectional, validated link map ready for construction of the linear program
    """

    # Work on copies to avoid mutating caller data
    private_df = private_links.copy()
    public_df = public_links.copy()
    devices_df = devices.copy()

    # Cast operators as strings and add to private links
    devices_df["Operator"] = devices_df["Operator"].astype(str)
    private_df = private_df.merge(devices_df[["Device", "Operator"]], left_on="Device1", right_on="Device",
                                  how="left").rename(columns={"Operator": "Operator1"}).drop(columns="Device")
    private_df = private_df.merge(devices_df[["Device", "Operator"]], left_on="Device2", right_on="Device",
                                  how="left").rename(columns={"Operator": "Operator2"}).drop(columns="Device")

    # Duplicate devices with outbound flag
    devices_df["Outbound"] = False
    outbound = devices_df.copy()
    outbound["Outbound"] = True
    devices_df = pd.concat([devices_df, outbound], ignore_index=True)

    # Duplicate private links, so matrix represents one-way flows only between switches
    max_shared = int(private_df["Shared"].max(skipna=True)) if pd.notna(private_df["Shared"].max(skipna=True)) else 0
    rev = private_df.copy()
    rev[["Device1", "Device2"]] = rev[["Device2", "Device1"]]
    rev[["Operator1", "Operator2"]] = rev[["Operator2", "Operator1"]]
    rev["Shared"] = rev['Shared'] + max_shared
    private_df = pd.concat([private_df, rev], ignore_index=True)

    # Adjust private bandwidth for uptime and make private links available to all traffic types
    private_df["Bandwidth"] *= private_df["Uptime"]
    private_df["Type"] = 0

    # Compact down shared IDs (if gaps in series) for private links
    max_shared = int(private_df["Shared"].max(skipna=True)) if pd.notna(private_df["Shared"].max(skipna=True)) else 0
    na_shared = private_df["Shared"].isna()
    if na_shared.any():
        private_df.loc[na_shared, "Shared"] = np.arange(max_shared + 1, max_shared + 1 + na_shared.sum())
    private_df["Shared"] = _unique_int(private_df["Shared"])

    # Rename public cities into switches
    public_df["City1"] = public_df["City1"].astype(str) + "00"
    public_df["City2"] = public_df["City2"].astype(str) + "00"

    # Duplicate public links, so matrix represents one-way flows only
    rev_public = public_df.copy()
    rev_public[["City1", "City2"]] = rev_public[["City2", "City1"]]
    public_df = pd.concat([public_df, rev_public], ignore_index=True)
    public_df["Type"] = 0

    # Add shared variables for edges
    devices_df["Shared"] = np.arange(private_df["Shared"].max() + 1, private_df["Shared"].max() + 1 + len(devices_df))

    # For demand starting and ending points, add direct on-ramps to public and private networks alike
    ramps_public, ramps_private = [], []
    for t in demand["Type"].unique():
        src  = demand.loc[demand["Type"] == t, "Start"].iat[0]
        dsts = demand.loc[demand["Type"] == t, "End"].unique()

        # Create public on-ramp for source and off-ramps for each destination
        ramps_public.append(dict(City1 = src, City2 = src + "00", Latency = 0, Type = t))
        for dst in dsts:
            ramps_public.append(dict(City1 = dst + "00", City2 = dst, Latency = 0, Type = t))

        # Create private on-ramps for each switch in source's city
        for _, r in devices_df[(devices_df["Device"].str[:3] == src) & (~devices_df["Outbound"])].iterrows():
            ramps_private.append(dict(Device1 = src, Device2 = r["Device"], Latency = 0, Bandwidth = r["Edge"],
                                      Operator1 = r["Operator"], Operator2 = r["Operator"], Shared = r["Shared"],
                                      Type = t))

        # Create private off-ramps for each switch in each destination's city
        for dst in dsts:
            for _, r in devices_df[(devices_df["Device"].str[:3] == dst) & (devices_df["Outbound"])].iterrows():
                ramps_private.append(dict(Device1 = r["Device"], Device2 = dst, Latency = 0, Bandwidth = r["Edge"],
                                          Operator1 = r["Operator"], Operator2 = r["Operator"], Shared = r["Shared"],
                                          Type = t))

    public_df = pd.concat([public_df, pd.DataFrame(ramps_public)], ignore_index=True)
    private_df = pd.concat([private_df, pd.DataFrame(ramps_private)], ignore_index=True)

    # For cities with both private/public links, create crossover points where contiguity_bonus is levied
    crossover_points = []
    crossover_cities = np.intersect1d(devices_df["Device"].str[:3], public_df["City1"].str[:3])
    for city in crossover_cities:
        # Create pathways from switches to public nodes
        for _, r in devices_df[(devices_df["Device"].str[:3] == city) & (devices_df["Outbound"])].iterrows():
            crossover_points.append(dict(Device1 = r["Device"], Device2 = city + "00", Latency = contiguity_bonus,
                                         Bandwidth = r["Edge"], Operator1 = r["Operator"], Operator2 = r["Operator"],
                                         Shared = r["Shared"], Type = 0))
        # Create pathways from public nodes to switches
        for _, r in devices_df[(devices_df["Device"].str[:3] == city) & (~devices_df["Outbound"])].iterrows():
            crossover_points.append(dict(Device1 = city + "00", Device2 = r["Device"], Latency = contiguity_bonus,
                                         Bandwidth = r["Edge"], Operator1 = r["Operator"], Operator2 = r["Operator"],
                                         Shared = r["Shared"], Type = 0))
    if len(crossover_points) > 0:
        private_df = pd.concat([private_df, pd.DataFrame(crossover_points)], ignore_index=True)

    # Decorate public links
    public_df = public_df.rename(columns={"City1": "Device1", "City2": "Device2"})
    public_df["Bandwidth"] = 0
    public_df["Operator1"] = "Public"
    public_df["Operator2"] = "Public"
    public_df["Shared"] = 0

    # Return fully consolidated map of private and public links
    return pd.concat(
        [private_df, public_df],
        ignore_index=True
    )

def lp_primitives(
    link_df: pd.DataFrame,
    demand_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Translate link map and demand into the core linear program primitives

    Parameters
    ----------
    link_df : pandas.DataFrame
        Full link table (private, public, helper, etc)
        `[Device1, Device2, Latency, Bandwidth, Uptime, Shared, Operator1, Operator2, Type]`
    demand_df : pandas.DataFrame
        Demand matrix `[Start, End, Receivers, Traffic, Priority, Type, Multicast, Original]`

    Returns dict with keys:
        A_eq         : csr_matrix – equality constraint matrix (flow)
        A_ub         : csr_matrix - inequality constraint matrix (bandwidth)
        b_eq         : ndarray    – RHS vector (traffic requirements)
        b_ub         : ndarray    – RHS vector (bandwidth limitations)
        cost         : ndarray    – objective coefficients (latencies)
        row_index1/2 : ndarray    – operator tags per row for inequality constraint matrix (0 = none)
        col_index1/2 : ndarray    – operator tags per column for all matrices (0 = none)
    """

    # Do booking keeping on numbers of different link types and multicast-eligible links
    n_private = int((link_df["Operator1"] != "Public").sum())
    mcast_eligible = link_df.index[~(link_df["Device2"].str[3:] == "00") & ~(link_df["Device2"].str[3:] == "") &
                                   (link_df["Operator1"] != "Public")].to_numpy()
    mcast_ineligible = link_df.index[((link_df["Device2"].str[3:] == "00") | (link_df["Device2"].str[3:] == "")) &
                                     (link_df["Operator1"] != "Public")].to_numpy()
    n_links = len(link_df)

    # Enumerate all nodes with indices
    nodes = np.sort(pd.unique(np.concatenate([link_df["Device1"], link_df["Device2"],
                                              demand_df["Start"], demand_df["End"]])))
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Build constraint matrix for links, on node x edge matrix
    rows, cols, data = [], [], []
    for j, (s, e) in link_df[["Device1", "Device2"]].iterrows():
        rows += [node_idx[s], node_idx[e]]
        cols += [j, j]
        data += [1, -1]
    A_single = csr_matrix((data, (rows, cols)), shape=(len(nodes), n_links))

    # Enumerate commodities and adjacent information
    commodities = np.sort(demand_df["Type"].unique())
    k_of_type = {t: k for k, t in enumerate(commodities)}
    commodity_multicast_flag = demand_df.groupby("Type")["Multicast"].first().to_dict()
    multicast_commodities = np.sort(demand_df.loc[demand_df["Multicast"], "Original"].unique())

    # Replicate constraint matrix for each traffic commodity via block diagonal
    A = block_diag([A_single] * len(commodities), format="csr")

    # Certain edges can only be traversed by certain traffic types; remove incorrect matches
    keep: List[int] = []
    for k, t in enumerate(commodities):
        valid = np.where((link_df["Type"] == t) | (link_df["Type"] == 0))[0]
        keep.extend(valid + k * n_links) # do offset since traffic types block diagonal
    keep = np.asarray(keep)

    # Define bandwidth constraints that account for shared bandwidth and multicast

    # Create building block matrix that tracks groups and links
    J1 = csr_matrix((np.ones(n_private), (link_df.loc[:n_private - 1, "Shared"] - 1, np.arange(n_private))),
                    shape=(link_df["Shared"].max(), n_links))
    J2 = csr_matrix((np.ones(len(mcast_ineligible)),
                     (link_df.loc[mcast_ineligible, "Shared"] - 1, mcast_ineligible)),
                    shape=(link_df["Shared"].max(), n_links))
    K  = csr_matrix((np.ones(len(mcast_eligible)), (np.arange(len(mcast_eligible)), mcast_eligible)),
                    shape=(len(mcast_eligible), n_links))

    # Create core constraints on bandwidth
    I_blocks = []
    for t in commodities:
        if commodity_multicast_flag.get(t):
            I_blocks.append(J2)
        else:
            I_blocks.append(J1)
    I = sp_hstack(I_blocks, format="csr") if n_private > 0 else csr_matrix((0, A.shape[1]))

    # Extend to multicast
    if multicast_commodities.size:

        # Extend constraint matrix for each multicast group
        I = sp_hstack([I, sp_hstack([(J1 - J2)[:, mcast_eligible]] * len(multicast_commodities), format="csr")],
                      format="csr")

        # Add rows for within-group multicast constraints
        for t in commodities:
            if not commodity_multicast_flag.get(t):
                continue

            # Multicast replicates traffic to multiple receivers
            receivers = float(demand_df.loc[demand_df["Type"] == t, "Receivers"].iloc[0])
            multicast_group = int(demand_df.loc[demand_df["Type"] == t, "Original"].iloc[0])
            vec1 = np.zeros(len(commodities))
            vec1[k_of_type[t]] = 1.0 / receivers
            vec2 = np.zeros(len(multicast_commodities))
            vec2[np.where(multicast_commodities == multicast_group)[0][0]] = -1.0

            # Add constraints to growing I matrix
            left = sp_hstack([K * v for v in vec1], format="csr")
            right = sp_hstack([K[:, mcast_eligible] * v for v in vec2], format="csr")
            I = sp_vstack([I, sp_hstack([left, right], format="csr")], format="csr")

        # Pad columns to keep
        keep_extension = n_links * len(commodities) + np.arange(len(multicast_commodities) * len(mcast_eligible))
        keep = np.concatenate((keep, keep_extension), axis = 0)

        # Pad the A matrix to match the size of the I matrix
        A = sp_hstack([A, csr_matrix((A.shape[0], I.shape[1] - A.shape[1]))], format="csr")

    A = A[:, keep]
    I = I[:, keep]

    # Build RHS vector of traffic requirements
    b_flows: List[NDArray] = []
    for t in commodities:
        vec = np.zeros(len(nodes))
        for _, r in demand_df[demand_df["Type"] == t].iterrows():
            qty = float(r["Traffic"]) * float(r["Receivers"])
            vec[node_idx[r["Start"]]] += qty
            vec[node_idx[r["End"]]]   -= qty
        b_flows.append(vec)
    b = np.concatenate(b_flows)

    # Build RHS vector of bandwidth limitations
    sorted_dupes = link_df.iloc[: n_private].sort_values("Shared").drop_duplicates("Shared")
    cap = sorted_dupes["Bandwidth"].to_numpy()
    cap = np.concatenate((cap, np.zeros(demand_df["Multicast"].sum() * len(mcast_eligible))), axis = 0)

    # Note which rows in bandwidth matrix are owned by which operators (Operator1 and Operator2)
    row_op1_multicast = _rep(link_df["Operator1"].iloc[mcast_eligible].to_numpy(), demand_df["Multicast"].sum())
    row_op1 = np.concatenate((sorted_dupes["Operator1"].to_numpy(), row_op1_multicast), axis = 0)
    row_op2_multicast = _rep(link_df["Operator2"].iloc[mcast_eligible].to_numpy(), demand_df["Multicast"].sum())
    row_op2 = np.concatenate((sorted_dupes["Operator2"].to_numpy(), row_op2_multicast), axis = 0)

    # Note which edges in all matrices are owned by which operators (Operator1 and Operator2)
    col_op1 = _rep(link_df["Operator1"].to_numpy(), len(commodities))
    col_op1_multicast =_rep(link_df["Operator1"].iloc[mcast_eligible].to_numpy(), len(multicast_commodities))
    col_op1 = np.concatenate((col_op1, col_op1_multicast), axis = 0)[keep]
    col_op2 = _rep(link_df["Operator2"].to_numpy(), len(commodities))
    col_op2_multicast =_rep(link_df["Operator2"].iloc[mcast_eligible].to_numpy(), len(multicast_commodities))
    col_op2 = np.concatenate((col_op2, col_op2_multicast), axis = 0)[keep]

    # Build objective function coefficients (latency multiplied by average priority)
    cost = _rep(link_df["Latency"].astype(float).to_numpy(), len(commodities))
    cost *= np.repeat(demand_df.groupby("Type")["Priority"].mean().loc[commodities].to_numpy(), len(link_df))
    cost = np.concatenate((cost, np.zeros(len(multicast_commodities) * len(mcast_eligible))), axis = 0)[keep]

    # Return all primitives as dictionary
    return dict(
        A_eq=A,
        A_ub=I,
        b_eq=b,
        b_ub=cap,
        cost=cost,
        row_index1=row_op1,
        row_index2=row_op2,
        col_index1=col_op1,
        col_index2=col_op2
    )

def network_shapley(
    private_links: pd.DataFrame,
    devices: pd.DataFrame,
    demand: pd.DataFrame,
    public_links: pd.DataFrame,
    operator_uptime: float = 1.0,
    contiguity_bonus: float = 5.0,
    demand_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Compute Shapley values per operator

    Parameters
    ----------
    private_links : pandas.DataFrame
        Private link table `[Start, End, Latency, Bandwidth, Uptime, Shared]`
    devices : pandas.DataFrame
        Devices table `[Device, Edge, Operator]`
    demand : pandas.DataFrame
        Demand matrix `[Start, End, Receivers, Traffic, Priority, Type, Multicast]`
    public_links : pandas.DataFrame
        Public internet links `[Start, End, Latency]`
    operator_uptime : float
        Reliability (between 0 - 1) of an operator in any given epoch
    contiguity_bonus : float
        Extra latency effectively added for mixing public links with private links
    demand_multiplier: float
        Extra multiplier to scale up demand

    Returns pandas.DataFrame
        Value and percent of value ascribed to each operator in simulation
    """

    # Check integrity in inputs
    check_inputs(private_links, devices, demand, public_links, operator_uptime)

    # Get consolidated map of links and adjusted demand
    full_demand = consolidate_demand(demand, demand_multiplier)
    full_map = consolidate_links(private_links, devices, full_demand, public_links, contiguity_bonus)

    # Construct linear program for analysis
    prim = lp_primitives(full_map, full_demand)

    # Enumerate all operators (except Private/Public tags)
    operators = np.sort([x for x in pd.unique(devices["Operator"].dropna().astype(str)) if x != 'Private'])
    n_ops = len(operators)

    # Construct coalitions bitmap: bitmap[i, j] = 1 iff operator i is in coalition j
    bitmap = _bits(n_ops)

    # Setup vectors to record results
    n_coal = 2 ** n_ops
    svalue = np.full(n_coal, -np.inf)
    size = np.zeros(n_coal, dtype=int)

    # Iterate over coalitions and solve linear program for each set of operators and their links (plus public links)
    for idx in range(n_coal):
        subset = operators[bitmap[:, idx] == 1]
        size[idx] = subset.size

        # Masks used to access relevant coalition sets (and public operator)
        row_mask = (np.isin(prim["row_index1"], np.concatenate((["Public", "Private"], subset))) &
                    np.isin(prim["row_index2"], np.concatenate((["Public", "Private"], subset))))
        col_mask = (np.isin(prim["col_index1"], np.concatenate((["Public", "Private"], subset))) &
                    np.isin(prim["col_index2"], np.concatenate((["Public", "Private"], subset))))

        # Solve linear program and save result
        res = linprog(prim["cost"][col_mask],
                      A_ub=prim["A_ub"][row_mask][:, col_mask],
                      b_ub=prim["b_ub"][row_mask],
                      A_eq=prim["A_eq"][:, col_mask],
                      b_eq=prim["b_eq"],
                      bounds=(0, None),
                      method="highs"
        )
        if res.success:
            svalue[idx] = -res.fun  # negative to turn min objective into max objective

    # Compute the expected value (inclusive of downtime) for a given operator set
    if operator_uptime <= 0.99999999:

        # Build lower-triangle submask of whether a coalition i is a subset of coalition j
        submask = (bitmap[:, None, :] <= bitmap[:, :, None]).all(axis=0)
        submask &= np.tri(n_coal, dtype=bool)

        # Build a base probability matrix and cast it across submask
        base_p = operator_uptime ** size
        bp_masked = base_p * submask

        # Compute recursive coefficient matrix used for probability estimates
        coef = csr_matrix((1, 1), dtype=int)
        for i in range(n_ops):
            sz = 2 ** i
            top = sp_hstack([coef, csr_matrix((sz, sz), dtype=int)])
            bottom = sp_hstack([-coef - diags([1]*sz, format="csr"), coef])
            coef = sp_vstack([top, bottom], format="csr").astype(int)
            coef.eliminate_zeros()

        # Get the dense copy once and place them in contiguous memory after slicing
        coef_dense = coef.toarray()
        term = bp_masked @ (coef_dense * submask)
        part = (bp_masked + term) * submask

        # Compute dot product for every coalition at once (with special-case value)
        evalue = (svalue * part).sum(axis=1)
        evalue[0] = svalue[0]
    else:
        # Skip computations if operator_uptime = 1
        evalue = svalue

    # Compute per-operator Shapley value by comparing coalitions with/without operator
    shapley = np.zeros(n_ops)
    fact_n = math.factorial(n_ops)
    for k, op in enumerate(operators):
        with_op = np.where(bitmap[k] == 1)[0] # coalitions with operator
        without_op = with_op - (1 << k) # bitshift for coalitions without operator
        w = _fact(size[with_op] - 1) * _fact(n_ops - size[with_op]) / fact_n # do weight calculation
        shapley[k] = np.sum(w * (evalue[with_op] - evalue[without_op]))

    # Cast into percentages
    percent = np.maximum(shapley, 0)
    percent = percent / percent.sum() if percent.sum() > 0 else percent

    # Return results
    return pd.DataFrame({
        "Operator": operators,
        "Value": np.round(shapley, 4),
        "Percent": np.round(percent, 4),
    })
