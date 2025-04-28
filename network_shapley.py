# Packages
from __future__ import annotations
import math
from typing import List, Dict
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

def consolidate_map(
    private_links: pd.DataFrame,
    demand: pd.DataFrame,
    public_links: pd.DataFrame,
    hybrid_penalty: float,
) -> pd.DataFrame:
    """
    Construct a single and fully-validated link table, using both private and public links, for lp_primitives()

    Parameters
    ----------
    private_links : pandas.DataFrame
        Private link table `[Start, End, Cost, Bandwidth, Operator1, Operator2, Uptime, Shared]`
    demand : pandas.DataFrame
        Demand matrix `[Start, End, Traffic, Type]`
    public_links : pandas.DataFrame
        Public internet links `[Start, End, Cost]`
    hybrid_penalty : float
        Extra latency effectively added for mixing public links with private links

    Returns pandas.DataFrame
        A fully‑expanded, bidirectional, validated link map ready for construction of the linear program
    """

    # Work on copies to avoid mutating caller data
    private_df = private_links.copy()
    public_df = public_links.copy()
    demand_df = demand.copy()

    # Perform basic sanity checks on map length, switch names, and node names
    _assert(private_df.shape[0] > 0,
            "There must be at least one private link for this simulation.")
    _assert(_has_digit(private_df["Start"]).all() & _has_digit(private_df["End"]).all(),
            "Switches are not labeled correctly in private links; they should be denoted with an integer.")
    _assert(_has_digit(public_df["Start"]).all() & _has_digit(public_df["End"]).all(),
            "Switches are not labeled correctly in private links; they should be denoted with an integer.")
    _assert((~_has_digit(demand_df["Start"])).all() & (~_has_digit(demand_df["End"])).all(),
            "Endpoints are not labeled correctly in the demand matrix; they should not have an integer.")

    # Cast operators as strings and fill in any missing secondary operators
    private_df["Operator1"] = private_df["Operator1"].astype(str)
    private_df["Operator2"] = (private_df["Operator2"].fillna(private_df["Operator1"])).astype(str)

    # Duplicate private links, so matrix represents one-way flows only between switches
    max_shared = int(private_df["Shared"].max(skipna=True)) if pd.notna(private_df["Shared"].max(skipna=True)) else 0
    rev = private_df.copy()
    rev[["Start", "End"]] = rev[["End", "Start"]]
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

    # Perform sanity check on traffic type in demand matrix
    _assert((demand_df.groupby("Type")["Start"].nunique() == 1).all(),
            "All traffic of a single type must have a single source.")

    # Duplicate public links, so matrix represents one-way flows only
    rev_public = public_df.copy()
    rev_public[["Start", "End"]] = rev_public[["End", "Start"]]
    public_df = pd.concat([public_df, rev_public], ignore_index=True)
    public_df["Type"] = 0

    # Perform sanity checks on the public links spanning private link routes and demand nodes
    _assert(pd.merge(private_df, public_df, on = ['Start', 'End']).shape[0] == private_df.shape[0],
            "The public pathway is not fully specified for all the switches.")
    city_pairs = public_df.assign(Start=public_df["Start"].str[:3], End=public_df["End"].str[:3])[["Start", "End"]]
    _assert(pd.merge(demand_df, city_pairs.drop_duplicates()).shape[0] == demand_df.shape[0],
            "The public pathway is not fully specified for the demand points.")

    # Build both helper links (node to switch) and direct public paths (node to node), per traffic type
    helper_frames: List[pd.DataFrame] = []
    for t in demand_df["Type"].unique():
        src_city = demand_df.loc[demand_df["Type"] == t, "Start"].iat[0]
        dst_cities = demand_df.loc[demand_df["Type"] == t, "End"].unique()

        # Find quickest direct switch-to-switch public latency between pair of cities
        helper_dir = public_df[(public_df["Start"].str[:3] == src_city) & (public_df["End"].str[:3].isin(dst_cities))]
        helper_dir = helper_dir.assign(Start=helper_dir["Start"].str[:3], End=helper_dir["End"].str[:3])
        helper_dir = helper_dir.groupby(["Start", "End"], as_index=False)["Cost"].min()
        helper_dir["Type"] = t

        # Create zero-cost helper links between nodes and all respective switches in that city
        src_switches = public_df.loc[public_df["Start"].str[:3] == src_city, "Start"].unique()
        helper_src = pd.DataFrame({"Start": src_city, "End": src_switches, "Cost": 0, "Type": t})
        dst_switches = public_df.loc[public_df["End"].str[:3].isin(dst_cities), "End"].unique()
        helper_dst = pd.DataFrame({"Start": dst_switches, "End": [s[:3] for s in dst_switches], "Cost": 0, "Type": t})

        helper_frames.append(pd.concat([helper_dir, helper_src, helper_dst], ignore_index=True))

    # Apply latency penalty to public links, as these will only be used for hybrid routing now
    public_df["Cost"] += hybrid_penalty

    # Merge public components into one and add empty columns
    public_df = pd.concat([public_df, pd.concat(helper_frames, ignore_index=True)], ignore_index=True)
    public_df = public_df.assign(Bandwidth=0, Operator1='0', Operator2='0', Uptime=1, Shared=0)[private_df.columns]

    # Return fully consolidated map of private and public links
    return pd.concat(
        [private_df, public_df],
        ignore_index=True
    )

def lp_primitives(
    link_map: pd.DataFrame,
    demand: pd.DataFrame,
    demand_multiplier: float,
) -> Dict[str, object]:
    """
    Translate link map and demand into the core linear program primitives

    Parameters
    ----------
    link_map : pandas.DataFrame
        Full link table (private, public, helper, etc)
        `[Start, End, Cost, Bandwidth, Operator1, Operator2, Uptime, Shared, Type]`
    demand : pandas.DataFrame
        Demand matrix `[Start, End, Traffic, Type]`
    demand_multiplier: float
        Extra multiplier to scale up demand

    Returns dict with keys:
        A_eq         : csr_matrix – equality constraint matrix (flow)
        A_ub         : csr_matrix - inequality constraint matrix (bandwidth)
        b_eq         : ndarray    – RHS vector (traffic requirements)
        b_ub         : ndarray    – RHS vector (bandwidth limitations)
        cost         : ndarray    – objective coefficients (latencies)
        row_index1/2 : ndarray    – operator tags per row for inequality constraint matrix (0 = none)
        col_index1/2 : ndarray    – operator tags per column for all matrices (0 = none)
    """

    # Count number of private and total links
    n_private = int((link_map["Operator1"] != "0").sum())
    n_links = len(link_map)

    # Enumerate all nodes with indices
    nodes = np.sort(pd.unique(np.concatenate([link_map["Start"], link_map["End"], demand["Start"], demand["End"]])))
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Build constraint matrix for links, on node x edge matrix
    rows, cols, data = [], [], []
    for j, (s, e) in link_map[["Start", "End"]].iterrows():
        rows += [node_idx[s], node_idx[e]]
        cols += [j, j]
        data += [1, -1]
    A_single = csr_matrix((data, (rows, cols)), shape=(len(nodes), n_links))

    # Replicate constraint matrix for each traffic commodity via block diagonal
    commodities = np.sort(demand["Type"].unique())
    A = block_diag([A_single] * len(commodities), format="csr")

    # Certain edges can only be traversed by certain traffic types; remove incorrect matches
    keep: List[int] = []
    for k, t in enumerate(commodities):
        valid = np.where((link_map["Type"] == t) | (link_map["Type"] == 0))[0]
        keep.extend(valid + k * n_links) # do offset since traffic types block diagonal
    keep = np.asarray(keep)
    A = A[:, keep]

    # Build RHS vector of traffic requirements
    b_flows: List[NDArray] = []
    for t in commodities:
        vec = np.zeros(len(nodes))
        sub = demand[demand["Type"] == t]
        for _, r in sub.iterrows():
            vec[node_idx[r["Start"]]] += r["Traffic"] * demand_multiplier
            vec[node_idx[r["End"]]]   -= r["Traffic"] * demand_multiplier
        b_flows.append(vec)
    b = np.concatenate(b_flows)

    # Build bandwidth constraint matrix, accounting for shared bandwidth
    shared_ids = link_map.loc[: n_private - 1, "Shared"].to_numpy()
    I_single = csr_matrix(
        (np.ones(n_private), (shared_ids - 1, np.arange(n_private))),
        shape=(shared_ids.max(), n_links),
    )
    I = sp_hstack([I_single] * len(commodities), format="csr")[:, keep]

    # Build RHS vector of bandwidth limitations
    sorted_dupes = link_map.iloc[: n_private].sort_values("Shared").drop_duplicates("Shared")
    cap = sorted_dupes["Bandwidth"].to_numpy()

    # Note which rows in bandwidth matrix are owned by which operators (Operator1 and Operator2)
    row_op1 = sorted_dupes["Operator1"].to_numpy()
    row_op2 = sorted_dupes["Operator2"].to_numpy()

    # Note which edges in all matrices are owned by which operators (Operator1 and Operator2)
    col_op1 = _rep(link_map["Operator1"].to_numpy(), len(commodities))[keep]
    col_op2 = _rep(link_map["Operator2"].to_numpy(), len(commodities))[keep]

    # Build objective function coefficients (latency)
    cost = _rep(link_map["Cost"].to_numpy(), len(commodities))[keep]

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
    demand: pd.DataFrame,
    public_links: pd.DataFrame,
    operator_uptime: float = 1.0,
    hybrid_penalty: float = 5.0,
    demand_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Compute Shapley values per operator

    Parameters
    ----------
    private_links : pandas.DataFrame
        Private link table `[Start, End, Cost, Bandwidth, Operator1, Operator2, Uptime, Shared]`
    demand : pandas.DataFrame
        Demand matrix `[Start, End, Traffic, Type]`
    public_links : pandas.DataFrame
        Public internet links `[Start, End, Cost]`
    operator_uptime : float
        Reliability (between 0 - 1) of an operator in any given epoch
    hybrid_penalty : float
        Extra latency effectively added for mixing public links with private links
    demand_multiplier: float
        Extra multiplier to scale up demand

    Returns pandas.DataFrame
        Value and percent of value ascribed to each operator in simulation
    """

    # Enumerate all operators
    operators = np.sort(pd.unique(np.concatenate([private_links["Operator1"].dropna().astype(str),
                                                  private_links["Operator2"].dropna().astype(str)])))
    n_ops = len(operators)
    _assert("0" not in operators, "0 is a protected keyword for operator names; choose another.")
    _assert(n_ops < 16, "There are too many operators; we limit to 15 to prevent the program from crashing.")

    # Construct coalitions bitmap: bitmap[i, j] = 1 iff operator i is in coalition j
    bitmap = _bits(n_ops)

    # Get underlying linear program primitives from consolidated map of links and scaled demand
    full_map = consolidate_map(private_links, demand, public_links, hybrid_penalty)
    prim = lp_primitives(full_map, demand, demand_multiplier)

    # Setup vectors to record results
    n_coal = 2 ** n_ops
    svalue = np.full(n_coal, -np.inf)
    size = np.zeros(n_coal, dtype=int)

    # Iterate over coalitions and solve linear program for each set of operators and their links (plus public links)
    for idx in range(n_coal):
        subset = operators[bitmap[:, idx] == 1]
        size[idx] = subset.size

        # Masks used to access relevant coalition sets (and public operator)
        row_mask = (np.isin(prim["row_index1"], np.concatenate((["0"], subset))) &
                    np.isin(prim["row_index2"], np.concatenate((["0"], subset))))
        col_mask = (np.isin(prim["col_index1"], np.concatenate((["0"], subset))) &
                    np.isin(prim["col_index2"], np.concatenate((["0"], subset))))

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
    coef_dense = coef.A
    term = bp_masked @ (coef_dense * submask)
    part = (bp_masked + term) * submask

    # Compute dot product for every coalition at once (with special-case value)
    evalue = (svalue * part).sum(axis=1)
    evalue[0] = svalue[0]

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
