# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:30:27 2023

@author: merkebud, ivespe

Intro script for Exercise 3 ("Scheduling flexibility resources") 
in specialization course module "Flexibility in power grid operation and planning" 
at NTNU (TET4565/TET4575) 

"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyomo.opt import SolverFactory
from pyomo.core import Var
import pyomo.environ as en
import time

#%% Read battery specifications
parametersinput = pd.read_csv('./battery_data.csv', index_col=0)
parameters = parametersinput.loc[1]

#Parse battery specification
capacity=parameters['Energy_capacity']
charging_power_limit=parameters["Power_capacity"]
discharging_power_limit=parameters["Power_capacity"]
charging_efficiency=parameters["Charging_efficiency"]
discharging_efficiency=parameters["Discharging_efficiency"]
#%% Read load demand and PV production profile data
testData = pd.read_csv('./profile_input.csv')

# Convert the various timeseries/profiles to numpy arrays
Hours = testData['Hours'].values
Base_load = testData['Base_load'].values
PV_prod = testData['PV_prod'].values
Price = testData['Price'].values

# Make dictionaries (for simpler use in Pyomo)
dict_Prices = dict(zip(Hours, Price))
dict_Base_load = dict(zip(Hours, Base_load))
dict_PV_prod = dict(zip(Hours, PV_prod))
# %% Task 1

print('--- Task 1: optimization battery ---')

# --- Build model + time step ---
T_list = list(Hours)
m = en.ConcreteModel()
m.T = en.Set(initialize=T_list, ordered=True)

# infer Δt [h] from Hours if possible (fallback = 1.0)
try:
    diffs = np.diff(Hours.astype(float))
    dt = float(np.min(diffs[diffs > 0])) if diffs.size and np.any(diffs > 0) else 1.0
except Exception:
    dt = 1.0

# --- PARAMETERS ---
m.price    = en.Param(m.T, initialize=dict_Prices,    within=en.Reals)  # €/kWh
m.base_load= en.Param(m.T, initialize=dict_Base_load, within=en.Reals)  # kW
m.pv       = en.Param(m.T, initialize=dict_PV_prod,   within=en.Reals)  # kW
m.dt       = en.Param(initialize=float(dt))                               # h

m.E_cap    = en.Param(initialize=float(capacity))               # kWh
m.P_chmax  = en.Param(initialize=float(charging_power_limit))   # kW
m.P_dmax   = en.Param(initialize=float(discharging_power_limit))# kW
m.eta_ch   = en.Param(initialize=float(charging_efficiency))    # virkningsgrad charging
m.eta_dis  = en.Param(initialize=float(discharging_efficiency)) # virkningsgrad discharging

# --- VARIABLES ---
m.charging         = en.Var(m.T, domain=en.NonNegativeReals)         # charge power [kW]
m.discharging      = en.Var(m.T, domain=en.NonNegativeReals)         # discharge power [kW]
m.g_import  = en.Var(m.T, domain=en.NonNegativeReals)         # grid import [kW]
m.g_export  = en.Var(m.T, domain=en.NonNegativeReals)         # grid export [kW]

# SoC indexed on T plus an initial slot (-1)
m.T_soc = en.Set(initialize=[-1] + T_list, ordered=True)
m.s     = en.Var(m.T_soc, domain=en.NonNegativeReals)          # SoC [kWh]

# --- OBJECTIVE: minimize energy cost over the horizon ---
def obj_rule(m):
    # cost = (imports - exports) * price * Δt
    return sum((m.g_import[t]*m.price[t] - m.g_export[t]*m.price[t]) * m.dt for t in m.T)
m.obj = en.Objective(rule=obj_rule, sense=en.minimize)

# --- CONSTRAINTS ---

# (1) Power balance (all kW)
def balance_rule(m, t):
    return m.pv[t] + m.g_import[t] + m.discharging[t] == m.base_load[t] + m.charging[t] + m.g_export[t]
m.balance = en.Constraint(m.T, rule=balance_rule)

# (2) SoC dynamics with efficiencies (kWh)
# predecessor mapping: previous time or -1 for the initial SoC
t_prev = {T_list[i]: (T_list[i-1] if i > 0 else -1) for i in range(len(T_list))}

def soc_dyn_rule(m, t):
    return m.s[t] == m.s[t_prev[t]] + (m.eta_ch*m.charging[t] - m.discharging[t]/m.eta_dis) * m.dt
m.soc_dyn = en.Constraint(m.T, rule=soc_dyn_rule)

# (3) Boundary conditions: SoC start/end at zero
m.s_init = en.Constraint(expr=m.s[-1] == 0.0)                 # SoC before first step
m.s_end  = en.Constraint(expr=m.s[T_list[-1]] == 0.0)         # SoC at end

# (4) Operating limits
def soc_cap_rule(m, t):
    return m.s[t] <= m.E_cap
m.soc_cap = en.Constraint(m.T_soc, rule=soc_cap_rule)

def ch_limit_rule(m, t):
    return m.charging[t] <= m.P_chmax
m.ch_limit = en.Constraint(m.T, rule=ch_limit_rule)

def dis_limit_rule(m, t):
    return m.discharging[t] <= m.P_dmax
m.dis_limit = en.Constraint(m.T, rule=dis_limit_rule)

