# -*- coding: utf-8 -*-
"""
Created on 2023-07-14
gi
@author: ivespe

Intro script for Exercise 2 ("Load analysis to evaluate the need for flexibility") 
in specialization course module "Flexibility in power grid operation and planning" 
at NTNU (TET4565/TET4575) 

"""

# %% Dependencies

import pandapower as pp
import pandapower.plotting as pp_plotting
import pandas as pd
import os
import load_scenarios as ls
import load_profiles as lp
import pandapower_read_csv as ppcsv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandapower.topology as top
import networkx as nx
import networkx as nx

# %% Define input data

# Location of (processed) data set for CINELDI MV reference system
# (to be replaced by your own local data folder)
path_data_set         = 'CINELDI_MV_reference_system_v_2023-03-06'

filename_load_data_fullpath = os.path.join(path_data_set,'load_data_CINELDI_MV_reference_system.csv')
filename_load_mapping_fullpath = os.path.join(path_data_set,'mapping_loads_to_CINELDI_MV_reference_grid.csv')

# Subset of load buses to consider in the grid area, considering the area at the end of the main radial in the grid
bus_i_subset = [90, 91, 92, 96]

# Assumed power flow limit in MW that limit the load demand in the grid area (through line 85-86)
P_lim = 0.637 


# Maximum load demand of new load being added to the system (random between 1 and 2)
scaling_factor = np.random.uniform(1, 2)
print(f"Scaling factor (scaling_factor) for new load: {scaling_factor:.3f}")

# Which time series from the load data set that should represent the new load
i_time_series_new_load = 90


# %% TASK 1 ##
print("\n--- TASK 1 ---")
net = ppcsv.read_net_from_csv(path_data_set, baseMVA=10)

eg_buses = list(net.ext_grid.bus.values) # list with slack buses /external grid buses
eg_buses = list(net.ext_grid.bus.values) # list with slack buses /external grid buses
assert len(eg_buses) >= 1, "No ext_grid found"
slack_bus = int(eg_buses[0])  # typically bus 0

pp.runpp(net, init="auto")

# --- Find actual start bus (slack/ext_grid) ---
if len(net.ext_grid) == 0:
    raise RuntimeError("No ext_grid found – cannot determine start bus.")
BUS_START = int(net.ext_grid.bus.iloc[0])

# --- Build graph of the grid (respect switches, include trafos) ---
G = top.create_nxgraph(net, respect_switches=True, include_trafos=True)

# Safety check
if BUS_START not in G:
    raise RuntimeError(f"Start bus {BUS_START} (ext_grid) not in graph. Available nodes: {sorted(G.nodes())[:10]} ...")

# --- If bus 96 exists, use it; otherwise choose the farthest leaf from the slack ---
CANDIDATE_END = 96
if CANDIDATE_END in G:
    BUS_END = CANDIDATE_END
else:
    # Farthest bus by hop count from the slack (typical feeder tail)
    lengths = nx.single_source_shortest_path_length(G, BUS_START)
    # Keep only buses present in net.bus.index (graph also has trafos ends etc., but that’s fine)
    BUS_END = max(lengths, key=lengths.get)

print(f"Using BUS_START={BUS_START}, BUS_END={BUS_END}")

# --- Get the path and plot voltage profile along it ---
path_nodes = nx.shortest_path(G, source=BUS_START, target=BUS_END)
v_pu_along_path = net.res_bus.loc[path_nodes, "vm_pu"].to_numpy()

x = np.arange(len(path_nodes))
plt.figure(figsize=(14, 6))
plt.figure(figsize=(14, 6))
plt.plot(x, v_pu_along_path, marker="o")
plt.xticks(x, path_nodes, rotation=45, ha='right')  # Rotate and right-align
plt.xticks(x, path_nodes, rotation=45, ha='right')  # Rotate and right-align
plt.xlabel(f"Bus along feeder ({BUS_START} → {BUS_END})")
plt.ylabel("Voltage [p.u.]")
plt.title("Voltage profile along main radial (base case)")
plt.grid(True)
plt.tight_layout()
#plt.show()


# --- Lowest voltage in the 'area of interest' ---
# If your sheet names buses 85–96 but those IDs don’t exist, choose the *last 12 buses* on this feeder as the area.
if all(b in net.bus.index for b in range(85, 97)):
    area_buses = list(range(85, 97))
else:
    tail_len = min(12, len(path_nodes))
    area_buses = path_nodes[-tail_len:]
    print(f"(IDs 85–96 not found) Using feeder tail as area: {area_buses}")

vmin_area = net.res_bus.loc[area_buses, "vm_pu"].min()
bus_vmin = net.res_bus.loc[area_buses, "vm_pu"].idxmin()
#print(f"Lowest voltage in area: {vmin_area:.4f} p.u. at bus {bus_vmin}")

# %% TASK 2
print("\n--- TASK 2 ---")

# ---- configuration
AREA_LOAD_BUSES = [90, 91, 92, 96]   # loads to scale
N_SAMPLES = 20                       # number of random scale factors
VMIN_LIMIT = 0.95                    # voltage limit (p.u.)

# ---- pick/confirm area for "minimum voltage in the area"
# prefer buses 85–96 if they all exist; else use tail of feeder (last 12 buses)
if all(b in net.bus.index for b in range(86, 97)):
    AREA_FOR_VMIN = list(range(86, 97))
else:
    # auto-tail
    if len(net.ext_grid) == 0:
        raise RuntimeError("No ext_grid found.")
    b0 = int(net.ext_grid.bus.iloc[0])
    G = top.create_nxgraph(net, respect_switches=True, include_trafos=True)
    lengths = nx.single_source_shortest_path_length(G, b0)
    tail = max(lengths, key=lengths.get)
    path_nodes = nx.shortest_path(G, b0, tail)
    AREA_FOR_VMIN = path_nodes[-min(12, len(path_nodes)):]
print(f"Area for min voltage: {AREA_FOR_VMIN}")

# ---- restrict to the buses that actually exist in this net
AREA_LOAD_BUSES = [b for b in AREA_LOAD_BUSES if b in net.bus.index]
if not AREA_LOAD_BUSES:
    raise RuntimeError("None of buses 90, 91, 92, 96 exist in this model.")

# ---- snapshot base load setpoints (so we can restore each iteration)
base_p = net.load["p_mw"].copy()
base_q = net.load["q_mvar"].copy()

# map bus -> list of load element indices
loads_by_bus = net.load.groupby("bus").groups
target_idxs = []
for b in AREA_LOAD_BUSES:
    target_idxs += list(loads_by_bus.get(b, []))
target_idxs = list(set(target_idxs))

# ---- generate random scale factors in [1, 2]
rng = np.random.default_rng(seed=42)      # fix seed for reproducibility; remove/modify if you like
scales = np.sort(rng.uniform(1.0, 2.0, size=N_SAMPLES))

# ---- run sweep
rows = []
for s in scales:
    # restore base
    net.load["p_mw"] = base_p
    net.load["q_mvar"] = base_q

    # scale P and Q proportionally for the four buses (keeps power factor)
    net.load.loc[target_idxs, "p_mw"] = base_p.loc[target_idxs] * s
    net.load.loc[target_idxs, "q_mvar"] = base_q.loc[target_idxs] * s

    # power flow
    pp.runpp(net, init="results")

    # build the row for the table: P90, P91, P92, P96, Sum, Vmin_area
    row = {"scale": float(s)}
    p_sum = 0.0
    for b in [90, 91, 92, 96]:
        p_b = float(net.load.loc[net.load.bus == b, "p_mw"].sum()) if b in net.bus.index else np.nan
        row[f"P{b}_MW"] = p_b
        if not np.isnan(p_b):
            p_sum += p_b
    row["P_area_sum_MW"] = p_sum

    vmin_area = float(net.res_bus.loc[AREA_FOR_VMIN, "vm_pu"].min())
    row["Vmin_area_pu"] = vmin_area

    rows.append(row)

results = pd.DataFrame(rows).sort_values("P_area_sum_MW").reset_index(drop=True)

# ---- show table (first rows) and optionally save
print("\n=== Task 2 table (first 10 rows) ===")
print(results.head(10).to_string(index=False))
# results.to_csv("task2_table.csv", index=False)  # uncomment to save

# ---- plot: minimum area voltage vs aggregated area demand
plt.figure()
plt.plot(results["P_area_sum_MW"], results["Vmin_area_pu"], marker="o")
plt.axhline(VMIN_LIMIT, linestyle="--")
plt.xlabel("Aggregated load in area (buses 90/91/92/96) [MW]")
plt.ylabel("Minimum voltage in area [p.u.]")
plt.title("Min voltage vs aggregated area demand (scaling factor 1→2)")
plt.grid(True)
plt.tight_layout()
#plt.show()

# ---- quick summary for report
vmin = results["Vmin_area_pu"].min()
pmax = results["P_area_sum_MW"].max()
n_viol = int((results["Vmin_area_pu"] < VMIN_LIMIT).sum())
print(f"\nMin voltage across sweep: {vmin:.4f} p.u.")
print(f"Max aggregated area demand across sweep: {pmax:.2f} MW")
print(f"Hours/points below {VMIN_LIMIT:.2f} p.u.: {n_viol} of {len(results)}")



# %% Task 3 ##
print("\n--- TASK 3 ---")
# Extract hourly load time series for a full year for all the load points in the CINELDI reference system
# (this code is made available for solving task 3)

load_profiles = lp.load_profiles(filename_load_data_fullpath)

# Get all the days of the year
repr_days = list(range(1,366))

# Get normalized load profiles for representative days mapped to buses of the CINELDI reference grid;
# the column index is the bus number (1-indexed) and the row index is the hour of the year (0-indexed)
profiles_mapped = load_profiles.map_rel_load_profiles(filename_load_mapping_fullpath,repr_days)

# Retrieve normalized load time series for new load to be added to the area
new_load_profiles = load_profiles.get_profile_days(repr_days)
new_load_time_series = new_load_profiles[i_time_series_new_load]*scaling_factor
new_load_time_series = new_load_profiles[i_time_series_new_load]*scaling_factor

# Calculate load time series in units MW (or, equivalently, MWh/h) by scaling the normalized load time series by the
# maximum load value for each of the load points in the grid data set (in units MW); the column index is the bus number
# (1-indexed) and the row index is the hour of the year (0-indexed)
load_time_series_mapped = profiles_mapped.mul(net.load['p_mw'])

# Visualize the first few rows of the DataFrame
print("\nLoad time series mapped (first 10 rows):")
print(load_time_series_mapped.head(10))
print(load_time_series_mapped.shape) 

area_buses = [90, 91, 92, 96]
aggregated_load_area = load_time_series_mapped[area_buses].sum(axis=1)
plt.figure(figsize=(12, 5))
plt.plot(aggregated_load_area, color='tab:blue')
plt.xlabel('Hour of the year')
plt.ylabel('Aggregated Load Demand (MW)')
plt.title('Aggregated Load Demand Time Series for Grid Area (Buses 90, 91, 92, 96)')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
