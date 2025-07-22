import re
from network_shapley import network_shapley

def retag_links(links: pd.DataFrame, operator_focus: str) -> pd.DataFrame:
    """
    Retags Operator so that methodology performs link-by-link calculations

    Parameters
    ----------
    links : pandas.DataFrame
        Full link table (private, public, helper, etc)
        `[Device1, Device2, Latency, Bandwidth, Uptime, Shared, Operator1, Operator2, Type]`
    operator_focus : str
        Operator name to focus on, in computing value of individual links

    Returns pandas.DataFrame
        The original link table but with Operator1/Operator2 modified
    """

    # Collapse non‑focus links into a general private category
    links.loc[~links["Operator1"].isin(["Public", operator_focus]), "Operator1"] = "Others"
    links.loc[~links["Operator2"].isin(["Public", operator_focus]), "Operator2"] = "Others"

    # Tag links that need to be processed and do not need to be processed
    links["tag"] = True
    links.loc[~(links["Operator1"].eq(operator_focus) | links["Operator2"].eq(operator_focus)),"tag"] = False

    # Iterate through all links that need to be processed
    pattern = re.compile(r"^[A-Z]{3}(([1-9][0-9]*)|(0[1-9]))$")
    counter = 0
    while links["tag"].any():
        idx = links.index[links["tag"].to_numpy()][0]
        d1, d2 = links.at[idx, "Device1"], links.at[idx, "Device2"]

        # If both Device1 and Device2 are real devices (3-letter city code followed by 1- or 2-digit integer, that isn't
        # 00), then find the symmetric links and retag the operators accordingly
        if pattern.match(d1) and pattern.match(d2):
            sym_idx = links.index[(links["Device1"] == d2) & (links["Device2"] == d1) &
                                  (links["Bandwidth"] == links.at[idx, "Bandwidth"]) &
                                  (links["Latency"] == links.at[idx, "Latency"])][0]
            counter += 1
            if links.at[idx, "Operator1"] == operator_focus:
                links.at[idx, "Operator1"] = str(counter)
                links.at[sym_idx, "Operator2"] = str(counter)
            if links.at[idx, "Operator2"] == operator_focus:
                links.at[idx, "Operator2"] = str(counter)
                links.at[sym_idx, "Operator1"] = str(counter)
            links.at[idx, "tag"] = False
            links.at[sym_idx, "tag"] = False

        # Otherwise (e.g. edge connections), tag into a fixed private pathway
        else:
            links.at[idx, "Operator1"] = "Private"
            links.at[idx, "Operator2"] = "Private"
            links.at[idx, "tag"] = False

    # Drop tag column and return links table
    links.drop(columns=["tag"], inplace=True)
    return links

def network_linkestimate(
    private_links: pd.DataFrame,
    devices: pd.DataFrame,
    demand: pd.DataFrame,
    public_links: pd.DataFrame,
    operator_focus: str,
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
    operator_focus : str
        Operator name to focus on, in computing value of individual links
    contiguity_bonus : float
        Extra latency effectively added for mixing public links with private links
    demand_multiplier: float
        Extra multiplier to scale up demand

    Returns pandas.DataFrame
        Value and percent of value estimated for each link in simulation
    """

    # Fix operator uptime
    operator_uptime = 1.0

    # Check that no overlapping shared-group links exist for focus operator
    temp = (private_links.merge(devices, left_on="Device1", right_on="Device").merge(devices, left_on="Device2",
                                                                                     right_on="Device"))
    _assert(~((temp.loc[(temp["Operator_x"] == operator_focus) | (temp["Operator_y"] == operator_focus), "Shared"].
               dropna().value_counts() > 1).any()), "Shared groups are not allowed for links by operator_focus.")

    # Check that there are no duplicate links
    private_links["key"] = private_links[["Device1", "Device2"]].apply(lambda x: tuple(sorted(x)), axis=1)
    _assert(~(private_links.duplicated(["key", "Bandwidth", "Latency"], keep=False).any()), "Duplicate links found.")
    private_links.drop(columns=["key"], inplace=True)

    # Check integrity in inputs
    check_inputs(private_links, devices, demand, public_links, operator_uptime)

    # Get consolidated map of links and adjusted demand
    full_demand = consolidate_demand(demand, demand_multiplier)
    full_map = consolidate_links(private_links, devices, full_demand, public_links, contiguity_bonus)

    # Modify operator schema to focus on links
    full_map = retag_links(full_map, operator_focus)

    # Construct linear program for analysis
    prim = lp_primitives(full_map, full_demand)

    # Enumerate all operators (except Private/Public tags)
    operators = np.sort(np.unique(np.concatenate((full_map["Operator1"].values, full_map["Operator2"].values))))
    operators = operators[~np.isin(operators, ["Private", "Public"])]
    n_ops = len(operators)
    _assert(n_ops < 21, "Too many links; we limit to 20 to prevent the program from crashing.")

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

    # The operator_uptime parameter is forced to one, so no need to compute expected values as distinct from scenarios
    evalue = svalue

    # Compute per-operator Shapley value by comparing coalitions with/without operator
    shapley = np.zeros(n_ops)
    fact_n = math.factorial(n_ops)
    for k, op in enumerate(operators):
        with_op = np.where(bitmap[k] == 1)[0] # coalitions with operator
        without_op = with_op - (1 << k) # bitshift for coalitions without operator
        w = _fact(size[with_op] - 1) * _fact(n_ops - size[with_op]) / fact_n # do weight calculation
        shapley[k] = np.sum(w * (evalue[with_op] - evalue[without_op]))

    # Prepare and return output
    drop_tags = {"Public", "Private", "Others"}
    schedule = full_map[["Device1", "Device2", "Bandwidth", "Latency", "Operator1", "Operator2"]].copy()
    schedule = schedule.loc[~(schedule["Operator1"].isin(drop_tags) & schedule["Operator2"].isin(drop_tags))]
    schedule = schedule.loc[schedule["Device1"] < schedule["Device2"]]
    schedule["Operator"] = np.where(schedule["Operator1"].isin(drop_tags), schedule["Operator2"], schedule["Operator1"])
    schedule = schedule.merge(pd.DataFrame({"Operator": operators.astype(str), "Value": shapley}),
                              on="Operator", how="left")
    schedule = schedule[["Device1", "Device2", "Bandwidth", "Latency", "Value"]]
    schedule["Percent"] = np.maximum(schedule["Value"], 0.0)
    schedule["Percent"] /= schedule["Percent"].sum()
    return schedule
