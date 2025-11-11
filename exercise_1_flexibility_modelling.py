"""
Created on 2024-08-10

@author: ivespe, merkebud

Script for Exercise 1 ("Modelling flexibility resources") in specialization 
course module "Flexibility in power grid operation and planning" at NTNU (TET4565/TET4575) 
"""

# %% Define dependencies and EWH model function 

from math import exp
import matplotlib.pyplot as plt
import numpy as np


def make_load_profile_ewh(time_steps,P,T,S,T_a,C,R,T_min,T_max,t_act,S_act):
    """
    Generate load time series for electric water heater (EWH).
    
    Inputs:
        time_steps: Number of time steps (minutes) (int)
        P: Initial load demand of EWH in kW
        T: Initial temperature of EWH in degrees Celsius
        S: Initial EWH status (1 is heating, 0 is not)
        T_a: Ambient temperature in degrees Celsius
        C: Thermal capacitance in kWh/deg C
        R: Thermal resistance in (degrees Celsius)/kW
        T_min: Minimum allowed water temperature in degrees Celsius
        T_max: Maximum allowed water temperature in degrees Celsius        
        t_act: Time of flexibility activation (minutes)
        S_act: EWH thermostat status after activating flexibility; 1 turns all EWHs on; 
            0 turns all EWHs off
    
    Outputs:
        P_list: Load demand time series of EWH, list with one element per time step
        T_list: Temperature time series of EWH in degrees Celsius, 
            with one element per time step
        S_list: EWH status time series, list with one element per time step    
    """

    P_list = [P]
    T_list = [T]
    S_list = [S]

    for t in range(1,time_steps):        
        T_prev = T
        S_prev = S

        # Solve differential equation for the change of temperature for the next time step
        T = T_a - exp(-(1/60)/(C*R))*(T_a + P_m * R * S_prev  - T_prev) + P_m * R * S_prev 

        if (T <= T_min) & (S_prev == 0):
            # Turn EWH on if the temperature becomes too low
            S = 1
        elif (T > T_max) & (S_prev == 1):
            # Turn EWH off if the temperature becomes too high
            S = 0
        else:
            S = S_prev

        if (t == t_act) & (S_act is not None):
            # Activate flexibility
            S = S_act

        # The EWH operates at full power capacity if turned on
        P = S_prev * P_m

        P_list.append(P)
        T_list.append(T)
        S_list.append(S)

    return P_list, T_list, S_list

# %% Initialize Electric Water Heater model

#Rated power of EWH in kW
P_m = 2

#Ambient temperature in degrees Celsius
T_a = 24

#Minimum allowed water temperature in degrees Celsius
T_min = 70

#Maximum allowed water temperature in degrees Celsius
T_max = 75

#Thermal capacitance in kWh/deg C
C = 0.335

#Thermal resistance in deg C /kW
R = 600

# Initializing EWH temperature (only applicable if modelling a single EWH)
T = 73

# Initialize electric water heater(s) to be turned off
S = 0
P = 0

# Time of flexibility activation (minutes from start time); 
# set to None to disable flexibility activation
t_act = 600

#600, 850, 1000, 1930, 2340

# EWH activation signal that sets the status of the EWHs after activating flexibility; 
# 1 turns all EWHs on; 0 turns all EWHs off; set to None to disable flexibility activation
S_act = 0

# Number of time steps (minutes)
time_steps = 48*60

# Number of EWHs / hot water tanks to model
N_EWH = 100

if N_EWH == 1:
    # If modelling a single EWH, initialize temperature as specified above
    T_init = [T]
elif N_EWH > 1:
    # If modelling multiple EWHs, initialize with random temperature
    rng = np.random.default_rng(seed=42)
    T_init = rng.uniform(70,75,N_EWH)

# Initialize time series for aggregated load demand of all the EWHs 
P_list_all = np.zeros(time_steps)

# Initialize time series for aggregated baseline load demand of all the EWHs, 
# i.e., the expected load demand without flexibility activation
P_list_base_all = np.zeros(time_steps)

# %% Run Electric Water Heater model

# Loop over all the EWHs
for i_EWH in range(N_EWH):
    T = T_init[i_EWH]
    
    # Trick to initialize EWH status randomly approximately according to steady-state distribution
    if i_EWH > 0:        
        S = S_list[-1]
        P = P_list[-1]
    
    # Run EWH model without flexibility activation to obtain the baseline power consumption pattern
    P_list_base, T_list_base, _ = make_load_profile_ewh(time_steps,P,T,S,T_a,C,R,T_min,T_max,None,None)
    
    # Run EWH model with flexibility activation
    P_list, T_list, S_list = make_load_profile_ewh(time_steps,P,T,S,T_a,C,R,T_min,T_max,t_act,S_act)
    
    # Aggregate load time series
    P_list_all += np.array(P_list)
    P_list_base_all += np.array(P_list_base)

#%% Plot results for from Electric Water Heater model

if N_EWH == 1:
    # If running model for a single Electric Water Heater
    fig,ax1 = plt.subplots()
    if (t_act != None) & (S_act != None):
        h_T_base, = plt.plot(T_list_base, 'r--')
    h_T, = plt.plot(T_list, 'r')
    ax1.set_ylim(ymin = T_min * 0.95)
    ax1.set_ylim(ymax = T_max * 1.05)
    color1 = 'tab:red'
    ax1.set_xlabel('minutes')
    ax1.set_ylabel('Temperature (degrees Celsius)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    if (t_act != None) & (S_act != None):
        ax1.legend([h_T_base,h_T], ['without flex.','with flex.'], loc = 'upper left')

    ax2 = ax1.twinx()  # instantiate a minute Axes that shares the same x-axis
    color2 = 'tab:blue'
    ax2.set_ylabel('Electric water heating power consumption (kW)', color=color2)  # we already handled the x-label with ax1
    ax2.plot(P_list_base, color=color2, linestyle='dashed')
    ax2.plot(P_list, color=color2)    
    ax2.tick_params(axis='y', labelcolor=color2)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

elif N_EWH > 1:
    # If running model for multiple Electric Water Heaters
    fig,ax1 = plt.subplots()
    ax1.set_ylim(ymin = 0)
    ax1.set_ylim(ymax = P_list_base_all.max() * 1.05)
    color1 = 'tab:blue'
    ax1.set_ylabel('Aggregated electric water heating power consumption (kW)', color=color1)
    if (t_act != None) & (S_act != None):
        h_P_base, = ax1.plot(P_list_base_all, color=color1, linestyle='dashed')
    h_P, = ax1.plot(P_list_all, color=color1)
    if (t_act != None) & (S_act != None):
        ax1.legend([h_P_base,h_P], ['without flex.','with flex.'], loc = 'upper left')
    plt.show()

P_flex = P_list_all - P_list_base_all   # flexibility in kW

plt.figure(figsize=(10,5))
plt.axhline(0, color='k', linestyle='--', linewidth=1)   # baseline reference
plt.plot(P_flex, color='tab:green')
plt.xlabel("Time (minutes)")
plt.ylabel("Flexibility (kW)")
plt.title("Activated Flexibility vs. Baseline Consumption")
plt.grid(True)
plt.show()



# # ---------------------------------------------------------------------------
# # Appendix (non-destructive): quantify flexibility for chosen activation times
# # This block leaves the existing model and variables intact and only appends
# # analysis for activation times [150, 175, 400, 1000]. It recomputes aggregated
# # profiles per activation time (so it does not modify earlier variables).
# # ---------------------------------------------------------------------------
# def _aggregate_population_for_activation(t_act_val, S_act_val):
#     """Aggregate baseline and with-activation profiles for the whole population.
#     Returns (P_base_total, P_with_total) arrays in kW."""
#     P_base_total = np.zeros(time_steps)
#     P_with_total = np.zeros(time_steps)
#     # loop over devices
#     for i in range(N_EWH):
#         T_dev = T_init[i]
#         # try to reuse last-device state if present, else default off
#         try:
#             prev_S = S_list[-1]
#             prev_P = P_list[-1]
#         except Exception:
#             prev_S = 0
#             prev_P = 0

#         # baseline for this device (no activation)
#         P_base_dev, _, _ = make_load_profile_ewh(time_steps, prev_P, T_dev, prev_S, T_a, C, R, T_min, T_max, None, None)
#         # with activation at t_act_val
#         P_with_dev, _, _ = make_load_profile_ewh(time_steps, prev_P, T_dev, prev_S, T_a, C, R, T_min, T_max, t_act_val, S_act_val)

#         P_base_total += np.array(P_base_dev)
#         P_with_total += np.array(P_with_dev)

#     return P_base_total, P_with_total


# activation_times = [150, 175, 400, 1000]
# S_act_used = 0
# summary_rows = []

# for t_act_val in activation_times:
#     P_base_tot, P_with_tot = _aggregate_population_for_activation(t_act_val, S_act_used)
#     P_flex = P_with_tot - P_base_tot

#     pos_capacity_kw = float(np.max(P_flex)) if P_flex.size else 0.0
#     neg_capacity_kw = float(-np.min(P_flex)) if P_flex.size else 0.0
#     pos_minutes = int((P_flex > 0).sum())
#     neg_minutes = int((P_flex < 0).sum())

#     summary_rows.append((t_act_val, pos_capacity_kw, neg_capacity_kw, pos_minutes, neg_minutes, float(np.max(P_base_tot)), float(np.max(P_with_tot))))

#     print(f"\n--- Activation t={t_act_val} min ---")
#     print(f"Peak increase (total): {pos_capacity_kw:.2f} kW, Peak reduction (total): {neg_capacity_kw:.2f} kW")
#     print(f"Positive activation duration: {pos_minutes} min ({pos_minutes/60.0:.2f} h)")
#     print(f"Negative activation duration: {neg_minutes} min ({neg_minutes/60.0:.2f} h)")

#     # plot aggregated baseline vs with-activation for this t_act
#     plt.figure(figsize=(10,3))
#     t = np.arange(len(P_with_tot))
#     plt.plot(t, P_base_tot, color='tab:blue', linestyle='--', label='Baseline')
#     plt.plot(t, P_with_tot, color='tab:blue', label='With activation')
#     plt.fill_between(t, P_base_tot, P_with_tot, where=(P_with_tot>P_base_tot), interpolate=True, color='tab:green', alpha=0.35)
#     plt.fill_between(t, P_base_tot, P_with_tot, where=(P_with_tot<P_base_tot), interpolate=True, color='tab:red', alpha=0.35)
#     plt.title(f'Aggregated EWH activation (t_act={t_act_val} min)')
#     plt.xlabel('Time (minutes)')
#     plt.ylabel('Aggregated power (kW)')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # Save summary CSV
# try:
#     import csv
#     csvfile = 'ewh_flex_summary.csv'
#     with open(csvfile, 'w', newline='') as f:
#         w = csv.writer(f)
#         w.writerow(['t_act_min','pos_capacity_kw','neg_capacity_kw','pos_minutes','neg_minutes','base_peak_kw','with_peak_kw'])
#         for r in summary_rows:
#             w.writerow(r)
#     print(f"\nSaved summary to {csvfile}")
# except Exception as e:
#     print("Could not write CSV summary:", e)

# # Overlay plot of aggregated flexibility (sampled every 5 minutes to reduce plot density)
# plt.figure(figsize=(12,4))
# sample = 5
# for (t_act_val, *_ ) in summary_rows:
#     P_base_tot, P_with_tot = _aggregate_population_for_activation(t_act_val, S_act_used)
#     P_flex = P_with_tot - P_base_tot
#     t = np.arange(0, len(P_flex), sample)
#     plt.plot(t, P_flex[t], label=f't={t_act_val} min')

# plt.axhline(0, color='k', linestyle='--', linewidth=0.6)
# plt.xlabel('Time (minutes)')
# plt.ylabel('Aggregated flexibility (kW)')
# plt.title('Aggregated EWH flexibility (overlay for activation times)')
# plt.legend(title='t_act')
# plt.grid(True)
# plt.tight_layout()
# plt.show()