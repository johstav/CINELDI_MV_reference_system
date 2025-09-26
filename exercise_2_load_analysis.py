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
 # provided module


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

# Maximum load demand of new load being added to the system
P_max_new = 0.4

# Which time series from the load data set that should represent the new load
i_time_series_new_load = 90


# %% Read pandapower network

net = ppcsv.read_net_from_csv(path_data_set, baseMVA=10)

eg_buses = list(net.ext_grid.bus.values)
assert len(eg_buses) >= 1, "No ext_grid found"
slack_bus = int(eg_buses[0])  # typically bus 0

# ---------- 2) Run a power flow ----------
pp.runpp(net, init="auto")

import pandapower.topology as top
import networkx as nx


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
plt.figure()
plt.plot(x, v_pu_along_path, marker="o")
plt.xticks(x, path_nodes, rotation=90)
plt.xlabel(f"Bus along feeder ({BUS_START} → {BUS_END})")
plt.ylabel("Voltage [p.u.]")
plt.title("Voltage profile along main radial (base case)")
plt.grid(True)
plt.tight_layout()
plt.show()

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
print(f"Lowest voltage in area: {vmin_area:.4f} p.u. at bus {bus_vmin}")

# %% Extract hourly load time series for a full year for all the load points in the CINELDI reference system
# (this code is made available for solving task 3)

load_profiles = lp.load_profiles(filename_load_data_fullpath)

# Get all the days of the year
repr_days = list(range(1,366))

# Get normalized load profiles for representative days mapped to buses of the CINELDI reference grid;
# the column index is the bus number (1-indexed) and the row index is the hour of the year (0-indexed)
profiles_mapped = load_profiles.map_rel_load_profiles(filename_load_mapping_fullpath,repr_days)

# Retrieve normalized load time series for new load to be added to the area
new_load_profiles = load_profiles.get_profile_days(repr_days)
new_load_time_series = new_load_profiles[i_time_series_new_load]*P_max_new

# Calculate load time series in units MW (or, equivalently, MWh/h) by scaling the normalized load time series by the
# maximum load value for each of the load points in the grid data set (in units MW); the column index is the bus number
# (1-indexed) and the row index is the hour of the year (0-indexed)
load_time_series_mapped = profiles_mapped.mul(net.load['p_mw'])
# %%
