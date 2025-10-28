# -*- coding: utf-8 -*-
"""
Created on 2023-10-10

@author: ivespe

Intro script for Exercise 4 ("Battery energy storage system in the grid vs. grid investments") 
in specialization course module "Flexibility in power grid operation and planning" 
at NTNU (TET4565/TET4575) 

"""


# %% Dependencies

import pandas as pd
import os
import load_profiles as lp
import pandapower_read_csv as ppcsv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# %% Define input data

# Location of (processed) data set for CINELDI MV reference system
# (to be replaced by your own local data folder)
path_data_set         = 'CINELDI_MV_reference_system_v_2023-03-06'

filename_load_data_fullpath = os.path.join(path_data_set,'load_data_CINELDI_MV_reference_system.csv')
filename_load_mapping_fullpath = os.path.join(path_data_set,'mapping_loads_to_CINELDI_MV_reference_grid.csv')
filename_standard_overhead_lines = os.path.join(path_data_set,'standard_overhead_line_types.csv')
filename_reldata = os.path.join(path_data_set,'reldata_for_component_types.csv')
filename_load_point = os.path.join(path_data_set,'CINELDI_MV_reference_system_load_point.csv')

# Subset of load buses to consider in the grid area, considering the area at the end of the main radial in the grid
bus_i_subset = [90, 91, 92, 96]

# Assumed power flow limit in MW that limit the load demand in the grid area (through line 85-86)
P_lim = 4

# Factor to scale the loads for this exercise compared with the base version of the CINELDI reference system data set
scaling_factor = 10

# Read standard data for overhead lines
data_standard_overhead_lines = pd.read_csv(filename_standard_overhead_lines, delimiter=';')
data_standard_overhead_lines.set_index(keys = 'type', drop = True, inplace = True)

# Read standard component reliability data
data_comp_rel = pd.read_csv(filename_reldata, delimiter=';')
data_comp_rel.set_index(keys = 'main_type', drop = True, inplace = True)

# Read load point data (incl. specific rates of costs of energy not supplied) for data
data_load_point = pd.read_csv(filename_load_point, delimiter=';')
data_load_point.set_index(keys = 'bus_i', drop = True, inplace = True)


# %% Read pandapower network

net = ppcsv.read_net_from_csv(path_data_set, baseMVA=10)


# %% Set up hourly normalized load time series for a representative day (task 2; this code is provided to the students)

load_profiles = lp.load_profiles(filename_load_data_fullpath)

# Consider only the day with the peak load in the area (28 February)
repr_days = [31+28]

# Get relative load profiles for representative days mapped to buses of the CINELDI test network;
# the column index is the bus number (1-indexed) and the row index is the hour of the year (0-indexed)
profiles_mapped = load_profiles.map_rel_load_profiles(filename_load_mapping_fullpath,repr_days)

# Calculate load time series in units MW (or, equivalently, MWh/h) by scaling the normalized load time series by the
# maximum load value for each of the load points in the grid data set (in units MW); the column index is the bus number
# (1-indexed) and the row index is the hour of the year (0-indexed)
load_time_series_mapped = profiles_mapped.mul(net.load['p_mw'])


# %% Aggregate the load demand in the area

# Aggregated load time series for the subset of load buses
load_time_series_subset = load_time_series_mapped[bus_i_subset] * scaling_factor
load_time_series_subset_aggr = load_time_series_subset.sum(axis=1)

P_max = load_time_series_subset_aggr.max()


# --- Task 1 ---


# %% Task 2: Planning horizon — peak growth (3% per year)
print('\n--- Task 2: Peak growth over 10-year planning horizon (3% p.a.) ---')

# base (reference) aggregated hourly series (MW)
base_aggr = load_time_series_subset_aggr.copy()

growth_rate = 0.03
# include year 10 (t = 10) so plots show the investment year at the end of the 10-year horizon
years = np.arange(0, 11)
peak_by_year = []
for y in years:
	factor = (1 + growth_rate) ** y
	peak_by_year.append((base_aggr * factor).max())

# Plot peaks vs year (step plot)
plt.figure(figsize=(8, 4))
plt.step(years, peak_by_year, where='post', linewidth=2)
# remove markers; show all years on the x-axis for clarity
plt.xticks(years)
plt.xlim(years.min(), years.max())
plt.axhline(P_lim, color='red', linestyle='--', label=f'Line flow limit P_lim = {P_lim:.2f} MW')
plt.xlabel('Year (y)')
plt.ylabel('Peak aggregated load (MW)')
plt.title('Peak aggregated load in area over 10-year horizon (3% annual growth)')
plt.grid(True)

# Find first year when peak exceeds limit
exceed_year = None
for y, peak in zip(years, peak_by_year):
	if peak > P_lim:
		exceed_year = int(y)
		break

if exceed_year is not None:
	plt.axvline(exceed_year, color='orange', linestyle=':', label=f'First exceed at y={exceed_year}')
	plt.annotate(f'First exceed at y={exceed_year}', xy=(exceed_year, peak_by_year[exceed_year]), xytext=(exceed_year+0.3, peak_by_year[exceed_year]+0.05*P_lim), arrowprops=dict(arrowstyle='->'))
	print(f"Peak aggregated load first exceeds P_lim={P_lim:.2f} MW at year y={exceed_year} (beginning of year {exceed_year+1}).")
else:
	print(f"Peak does not exceed P_lim={P_lim:.2f} MW within 10 years.")

plt.legend()
plt.tight_layout()
plt.show()

# %% Task 3: Estimate grid investment cost for alternative A (replace FeAl 35 -> FeAl 70)
print('\n--- Task 3: Estimate grid investment cost for alternative A ---')

km_of_new_line = 20
# DataFrame 'data_standard_overhead_lines' was read earlier and indexed by 'type'
old_type = 'FeAl 35'
new_type = 'FeAl 70'

old_type = 'FeAl 35'
new_type = 'FeAl 70'

# Helper to find best-matching row in the standard lines DataFrame
def find_best_type_match(df_index, target_keywords):
	# exact match first
	if target_keywords in df_index:
		return target_keywords
	# try to match by numeric gauge (e.g., '35' or '70')
	target_num = ''.join([c for c in target_keywords if c.isdigit()])
	if target_num:
		for idx in df_index:
			low = str(idx).lower()
			if target_num in low:
				return idx
	# fallback: match by 'feal' if requested
	if 'feal' in target_keywords.lower():
		for idx in df_index:
			low = str(idx).lower()
			if 'feal' in low:
				return idx
	return None

old_match = find_best_type_match(data_standard_overhead_lines.index, old_type)
new_match = find_best_type_match(data_standard_overhead_lines.index, new_type)

if old_match is None or new_match is None:
	print(f"Line types matching '{old_type}' or '{new_type}' not found in standard line data. Available example types: {list(data_standard_overhead_lines.index[:6])}...")
else:
	# Try to find a sensible cost column in the standard data
	cost_col = None
	for c in data_standard_overhead_lines.columns:
		if 'cost' in c.lower() or 'price' in c.lower() or 'investment' in c.lower() or 'capex' in c.lower():
			cost_col = c
			break

	if cost_col is None:
		print('No cost column found in data_standard_overhead_lines. Cannot estimate investment cost.')
	else:
		cost_old = data_standard_overhead_lines.loc[old_match, cost_col]
		cost_new = data_standard_overhead_lines.loc[new_match, cost_col]

		# Find the line(s) to be replaced: assume the critical line is between buses 85 and 86 (as in exercise description)
		# Search net.line for a line connecting those buses
		try:
			# net is available from earlier read
			mask = ((net.line.from_bus == 85) & (net.line.to_bus == 86)) | ((net.line.from_bus == 86) & (net.line.to_bus == 85))
			candidates = net.line[mask]
			if candidates.empty:
				# fallback: pick the longest line in the area tail (heuristic)
				if 'length_km' in net.line.columns:
					line_idx = net.line.index[net.line.length_km == net.line.length_km.max()][0]
					line_length_km = float(net.line.at[line_idx, 'length_km'])
				elif 'length' in net.line.columns:
					line_idx = net.line.index[0]
					line_length_km = float(net.line.at[line_idx, 'length'])
				else:
					line_idx = net.line.index[0]
					line_length_km = 0.1
				print('Could not find explicit line 85-86; using fallback line index', line_idx)
			else:
				# use first matching line
				line_idx = candidates.index[0]
				if 'length_km' in net.line.columns:
					line_length_km = float(net.line.at[line_idx, 'length_km'])
				elif 'length' in net.line.columns:
					# assume length is in km if named length
					line_length_km = float(net.line.at[line_idx, 'length'])
				else:
					# default length 0.1 km (fallback)
					line_length_km = 0.1

			# Compute investment cost
			# If the cost column is per km, multiply by length; if it's total cost per standard span, assume per km
			investment_old = float(cost_old) * line_length_km
			investment_new = float(cost_new) * line_length_km
			investment_delta = investment_new - investment_old
			present_value = investment_new*km_of_new_line

			print(f'Line types matched: old="{old_match}", new="{new_match}"')
			print(f'Line index replaced: {line_idx}, length = {line_length_km:.3f} km')
			print(f'Cost column used: {cost_col}')
			print(f'Estimated investment for new {new_match} line: {investment_new:,.2f} (same units as {cost_col})')
			print(f'Incremental investment (new - old): {investment_delta:,.2f}')
			print(f'Present value of 20 km: {present_value:,.2f}')


		except Exception as e:
			print('Error while estimating investment cost:', e)
 # %% Task 4: Present value calculation for the investment (discount rate)

    
	# Present value calculation for the investment (discount rate)

print('\n--- Task 4: Present value calculation for the investment (discount rate) ---')
		# Discount rate (annual)
discount_rate = 0.04
		# Investment implemented at beginning of year 2 -> time = 1 year from today (t=1)
investment_time_year = 1
pv_investment_new = investment_new / ((1 + discount_rate) ** investment_time_year)
pv_investment_delta = investment_delta / ((1 + discount_rate) ** investment_time_year)
present_value_new = pv_investment_new*km_of_new_line

print(f'\nPresent value (r={discount_rate*100:.1f}%) of investment for new line (paid at t={investment_time_year}): {pv_investment_new:,.2f}')
print(f'Present value (r={discount_rate*100:.1f}%) of incremental investment: {pv_investment_delta:,.2f}')
print(f'Present value of 20 km: {present_value_new:,.2f}')

# %% Task 5: Residual value and PV-corrected investment
print('\n--- Task 5: Residual value and PV-corrected investment ---')
# Economic assumptions
economic_life_years = 40
horizon_years = 20
# Investment timing: installed at beginning of year 2 -> t = 1
investment_time_year = 1

# Age of asset at end of horizon = horizon_years - investment_time_year
asset_age_at_horizon = horizon_years - investment_time_year
if asset_age_at_horizon < 0:
	asset_age_at_horizon = 0

# Linear depreciation: book value at horizon (per-km) = investment_new * (1 - age / life)
remaining_life_per_km = max(economic_life_years - asset_age_at_horizon, 0)
residual_value_per_km = investment_new * (remaining_life_per_km / economic_life_years)

# Scale to total project length
residual_value_total_nominal = residual_value_per_km * km_of_new_line

# Present value of residual received at end of horizon (t = horizon_years)
pv_residual_total = residual_value_total_nominal / ((1 + discount_rate) ** horizon_years)

# Present value of full new-line investment (already computed for total: present_value_new)
pv_investment_total = present_value_new

# PV-corrected investment (subtract PV of residual)
pv_corrected_investment_total = pv_investment_total - pv_residual_total

print(f'Asset age at horizon (years): {asset_age_at_horizon}')
print(f'Remaining life (years): {remaining_life_per_km} / {economic_life_years}')
print(f'Residual value (nominal) at t={horizon_years}: {residual_value_total_nominal:,.2f} (same units as {cost_col})')
print(f'Present value of residual (discounted to t=0, r={discount_rate*100:.1f}%): {pv_residual_total:,.2f}')
print(f'Present-value corrected investment (PV_investment_total - PV_residual_total): {pv_corrected_investment_total:,.2f}')


# %% Task 6: Battery postponement (1 MW) — deferred reinforcement and PV effect
print('\n--- Task 6: Battery postponement of reinforcement (1 MW resource) ---')
try:
	batt_power_mw = 1.0
	# peak_by_year contains the nominal peaks computed earlier
	peak_with_batt = [p - batt_power_mw for p in peak_by_year]

	# find first exceed year with battery
	exceed_year_batt = None
	for y, p in zip(years, peak_with_batt):
		if p > P_lim:
			exceed_year_batt = int(y)
			break

	if exceed_year_batt is None:
		print('Battery avoids congestion within the 10-year test horizon.')
	else:
		print(f'With 1 MW battery always available, first exceed of P_lim={P_lim:.2f} MW occurs at y={exceed_year_batt} (beginning of year {exceed_year_batt+1}).')

		# Plot peaks with battery (step plot) — similar style to Task 2
		plt.figure(figsize=(8, 4))
		plt.step(years, peak_with_batt, where='post', linewidth=2)
		plt.axhline(P_lim, color='red', linestyle='--', label=f'Line flow limit P_lim = {P_lim:.2f} MW')
		# mark the exceed year if present
		plt.axvline(exceed_year_batt, color='orange', linestyle=':', label=f'First exceed at y={exceed_year_batt}')
		plt.xlabel('Year (y)')
		plt.ylabel('Peak aggregated load with 1 MW battery (MW)')
		plt.title('Peak aggregated load with 1 MW battery — 10-year horizon')
		plt.xticks(years)
		plt.grid(True)
		plt.legend()
		plt.tight_layout()
		plt.show()

	# Investment parameters already defined above: investment_new (per-km), km_of_new_line, discount_rate,
	# economic_life_years, horizon_years
	if 'investment_new' not in globals():
		print('investment_new not available; cannot compute PV comparison for alternatives.')
	else:
		# helper functions
		def pv(amount, t):
			return amount / ((1 + discount_rate) ** t)

		# Alternative A (no battery): installation at t_A = investment_time_year (already set to 1)
		t_A = investment_time_year
		total_nominal = investment_new * km_of_new_line
		pv_inv_A = pv(total_nominal, t_A)
		# residual at horizon (we computed residual_value_total_nominal above)
		pv_resid_A = pv(residual_value_total_nominal, horizon_years)
		corrected_A = pv_inv_A - pv_resid_A

		# Alternative B (battery): installation at t_B = exceed_year_batt
		if exceed_year_batt is None:
			print('No reinforcement required under B within 10 years; skipping B calculations.')
		else:
			t_B = exceed_year_batt
			pv_inv_B = pv(total_nominal, t_B)
			# residual for B: remaining life = economic_life_years - (horizon_years - t_B)
			years_in_service_B = max(0, horizon_years - t_B)
			remaining_life_B = max(0, economic_life_years - years_in_service_B)
			residual_total_B = total_nominal * (remaining_life_B / economic_life_years)
			pv_resid_B = pv(residual_total_B, horizon_years)
			corrected_B = pv_inv_B - pv_resid_B

			# Print results
			print('\nAlternative A (reinforce at original exceed):')
			print(f'  install at t={t_A}, PV_inv = {pv_inv_A:,.2f}, PV_residual = {pv_resid_A:,.2f}, corrected PV = {corrected_A:,.2f}')

			print('\nAlternative B (1 MW battery available):')
			print(f'  install at t={t_B}, PV_inv = {pv_inv_B:,.2f}, PV_residual = {pv_resid_B:,.2f}, corrected PV = {corrected_B:,.2f}')

			savings = corrected_A - corrected_B
			print(f'\nDeferral savings (PV-corrected) A -> B: {savings:,.2f} (same units as above)')

except Exception as e:
	print('Error in Task 6:', e)

# %% Task 7: Annual operational costs of procuring battery congestion management
print('\n--- Task 7: Annual operational costs for battery congestion management (Alt B only) ---')
try:
	cost_per_MWh = 2000.0  # NOK per MWh
	representative_days_per_year = 20
	batt_power_mw = 1.0

	# reinforcement year for B
	reinf_B = exceed_year_batt if 'exceed_year_batt' in globals() and exceed_year_batt is not None else None

	# Exclude t=10 for Task 7 (use years 0..9)
	years_task7 = years[years < 10]
	annual_costs_B = []

	for y in years_task7:
		# scaled hourly series for the representative day
		factor = (1 + growth_rate) ** y
		daily_series = (base_aggr * factor)

		# hourly amount above limit
		hourly_over = (daily_series - P_lim).clip(lower=0)
		# battery can shift up to batt_power_mw each hour
		hourly_shift = np.minimum(hourly_over, batt_power_mw)
		daily_energy_shift = hourly_shift.sum()  # MWh for the representative day
		daily_cost = daily_energy_shift * cost_per_MWh
		annual_cost = daily_cost * representative_days_per_year

		# For alternative B: services procured until B's reinforcement
		if reinf_B is not None and y >= reinf_B:
			annual_costs_B.append(0.0)
		else:
			annual_costs_B.append(annual_cost)

	# Print results (B only) — exclude t=10
	print('\nYear | Annual cost B (NOK)')
	for y, b_cost in zip(years_task7, annual_costs_B):
		print(f'{y:>4} | {b_cost:>18,.2f}')

	# Plot B only
	plt.figure(figsize=(8,4))
	plt.step(years_task7, annual_costs_B, where='post', label='Annual battery cost — Alt B')
	plt.xlabel('Year (y)')
	plt.ylabel('Annual battery service cost (NOK)')
	plt.title('Annual operational costs for battery congestion management (Alt B)')
	plt.xticks(years_task7)
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.show()

except Exception as e:
	print('Error in Task 7:', e)


# %% Task 8: Estimate annual expected energy not supplied (EENS) for alternative A
print('\n--- Task 8: Annual expected energy not supplied (EENS) for alternative A ---')
try:
	# Parameters
	avg_load_year1_mw = 1.841  # given: average load demand in the area in year 1 (MW)
	length_km = km_of_new_line  # main feeder length = 20 km
	# get permanent failure rate and repair time for overhead lines from data_comp_rel
	comp_key = 'Overhead line (1–22 kV)'
	if comp_key not in data_comp_rel.index:
		print(f'Component reliability data for "{comp_key}" not found in data_comp_rel. Available keys: {list(data_comp_rel.index[:6])}...')
	else:
		lambda_perm = float(data_comp_rel.at[comp_key, 'lambda_perm'])
		r_perm = float(data_comp_rel.at[comp_key, 'r_perm'])  # repair time (hours)

		# Interpret lambda_perm as failures per 100 km-year (as in the dataset). Scale to feeder length.
		failures_per_year = lambda_perm * (length_km / 100.0)

		# Compute EENS for years 0..9 (exclude t=10 from calculations)
		years_task8 = years[years < 10]
		eens_by_year = []
		avg_loads = []
		for y in years_task8:
			# scale average load by growth
			avg_load = avg_load_year1_mw * ((1 + growth_rate) ** y)
			avg_loads.append(avg_load)
			# energy not supplied per failure (MWh) = avg_load (MW) * outage duration (h)
			ENS_per_failure = avg_load * r_perm
			eens = failures_per_year * ENS_per_failure  # MWh/year
			eens_by_year.append(eens)

		# Print table (exclude t=10)
		print('\nYear | EENS (MWh/year) | Average load (MW)')
		for y, eens, in_load in zip(years_task8, eens_by_year, avg_loads):
			print(f'{y:>4} | {eens:>14,.3f} | {in_load:>14.3f}')

		# Small plot for EENS — include t=10 only in plot by extending the series to t=10
		years_plot = np.append(years_task8, 10)
		# extend plot series by repeating last value at t=10 so the step plot shows the endpoint
		eens_plot = np.append(eens_by_year, eens_by_year[-1])
		plt.figure(figsize=(8,4))
		plt.step(years_plot, eens_plot, where='post', linewidth=2)
		plt.xlabel('Year (y)')
		plt.ylabel('EENS (MWh/year)')
		plt.title('Expected Energy Not Supplied (Alternative A)')
		plt.xticks(years_plot)
		plt.grid(True)
		plt.tight_layout()
		plt.show()

except Exception as e:
	print('Error in Task 8:', e)


# %% Task 9: Annual interruption costs (CENS) for alternative A
print('\n--- Task 9: Annual interruption costs (CENS) for alternative A ---')
try:
	# Use same years as Task 8 (exclude t=10 from calculations)
	years_task9 = years[years < 10]

	# Determine which cost column to use in data_load_point based on outage duration r_perm (hours)
	cost_cols = [c for c in data_load_point.columns if c.startswith('c_NOK_per_kWh_')]
	# parse hour number from column names like c_NOK_per_kWh_1h or _4h
	def col_hours(colname):
		s = colname.rsplit('_', 1)[-1]
		# remove trailing 'h' and convert
		try:
			return int(s.rstrip('h'))
		except:
			return None

	# find column with hours closest to r_perm
	col_diffs = {c: abs(col_hours(c) - r_perm) for c in cost_cols}
	selected_cost_col = min(col_diffs, key=col_diffs.get)

	# Build weighted average interruption cost (NOK/kWh) for the area buses
	area_buses = bus_i_subset
	# weights from network base loads (scaled)
	try:
		bus_loads = net.load.loc[area_buses, 'p_mw'] * scaling_factor
	except Exception:
		# fallback: equal weights
		bus_loads = pd.Series(1.0, index=area_buses)

	# get cost per bus; if missing use dataset mean
	cost_per_bus = []
	for b in area_buses:
		if b in data_load_point.index and pd.notna(data_load_point.at[b, selected_cost_col]):
			cost_per_bus.append(float(data_load_point.at[b, selected_cost_col]))
		else:
			cost_per_bus.append(float(data_load_point[selected_cost_col].mean()))

	bus_loads = bus_loads.reindex(area_buses).fillna(0.0)
	weights = bus_loads.values.astype(float)
	if weights.sum() == 0:
		weights = np.ones_like(weights)

	avg_cost_kNOK_per_kWh = float(np.dot(cost_per_bus, weights) / weights.sum())

	# Compute CENS for each year (exclude t=10)
	cens_by_year = []
	eens_calc = []
	for y in years_task9:
		avg_load = avg_load_year1_mw * ((1 + growth_rate) ** y)
		ENS_per_failure = avg_load * r_perm
		eens = failures_per_year * ENS_per_failure  # MWh/year
		eens_calc.append(eens)
		cens = eens * 1000.0 * avg_cost_kNOK_per_kWh  # NOK/year
		cens_by_year.append(cens)

	# Print table
	print(f"\nSelected interruption cost column: {selected_cost_col} (hours approx {col_hours(selected_cost_col)})")
	print('\nYear | EENS (MWh/yr) | Avg cost (NOK/kWh) | CENS (NOK/yr)')
	for y, eens, cens in zip(years_task9, eens_calc, cens_by_year):
		print(f'{y:>4} | {eens:>12,.3f} | {avg_cost_kNOK_per_kWh:>17,.3f} | {cens:>14,.2f}')

	# Plot CENS, include t=10 visually by extending last value
	years_plot = np.append(years_task9, 10)
	cens_plot = np.append(cens_by_year, cens_by_year[-1])
	plt.figure(figsize=(8,4))
	plt.step(years_plot, cens_plot, where='post', linewidth=2)
	plt.xlabel('Year (y)')
	plt.ylabel('Annual interruption cost CENS (NOK/year)')
	plt.title('Annual interruption costs (Alternative A)')
	plt.xticks(years_plot)
	plt.grid(True)
	plt.tight_layout()
	plt.show()

except Exception as e:
	print('Error in Task 9:', e)


# %% Task 10: Annual interruption costs (CENS) for alternative B (battery mitigates outages)
print('\n--- Task 10: Annual interruption costs (CENS) for alternative B ---')
try:
	# Use same years_task9 (0..9)
	years_task10 = years[years < 10]

	# Battery assumptions
	batt_energy_mwh = 2.0  # energy capacity available at start of outage (MWh)
	batt_power_mw = 1.0    # max power (MW)

	cens_B_by_year = []
	eens_B_by_year = []

	for y in years_task10:
		avg_load = avg_load_year1_mw * ((1 + growth_rate) ** y)
		ENS_no_batt = avg_load * r_perm  # MWh per failure without battery
		# battery can supply up to power * duration and energy capacity
		battery_supply_per_failure = min(batt_power_mw * r_perm, batt_energy_mwh, ENS_no_batt)
		ENS_with_batt = max(0.0, ENS_no_batt - battery_supply_per_failure)
		eens_B = failures_per_year * ENS_with_batt
		cens_B = eens_B * 1000.0 * avg_cost_kNOK_per_kWh  # NOK/year
		eens_B_by_year.append(eens_B)
		cens_B_by_year.append(cens_B)

	# Print table
	print('\nYear | EENS_B (MWh/yr) | CENS_B (NOK/yr)')
	for y, eens_b, cens_b in zip(years_task10, eens_B_by_year, cens_B_by_year):
		print(f'{y:>4} | {eens_b:>13,.3f} | {cens_b:>14,.2f}')

	# Plot CENS_B (include t=10 visually)
	years_plot = np.append(years_task10, 10)
	cens_B_plot = np.append(cens_B_by_year, cens_B_by_year[-1])
	plt.figure(figsize=(8,4))
	plt.step(years_plot, cens_B_plot, where='post', linewidth=2)
	plt.xlabel('Year (y)')
	plt.ylabel('Annual interruption cost CENS_B (NOK/year)')
	plt.title('Annual interruption costs with battery (Alternative B)')
	plt.xticks(years_plot)
	plt.grid(True)
	plt.tight_layout()
	plt.show()

except Exception as e:
	print('Error in Task 10:', e)


# %% Task 12: Build 10-year table (Years 1..10) for Alternative A and save CSV
print('\n--- Task 12: 10-year table for Alternative A  ---')
try:
	# Map script years y=0..9 to table Years 1..10
	table_years = list(range(1, 11))

	# Nominal total investment (undiscounted) occurs at t=1 -> table Year 2
	total_nominal = float(investment_new * km_of_new_line)

	# Recompute interruption costs for years y=0..9 to ensure availability
	cens_years_0_9 = []
	for y in range(0, 10):
		avg_load = avg_load_year1_mw * ((1 + growth_rate) ** y)
		ENS_per_failure = avg_load * r_perm
		eens = failures_per_year * ENS_per_failure
		cens = eens * 1000.0 * avg_cost_kNOK_per_kWh
		cens_years_0_9.append(cens)

	# Build rows
	rows = []
	for i, year in enumerate(table_years):
		y = year - 1
		inv = pv_corrected_investment_total if year == 2 else 0.0
		op_cost = 0.0  # Alternative A has no flexibility activation
		inter_cost = float(cens_years_0_9[y])
		rows.append({'Year': year, 'Investment cost (NOK)': inv, 'Operational costs (NOK)': op_cost, 'Interruption costs (NOK)': inter_cost})

	df_table = pd.DataFrame(rows)
	# Pretty print
	with pd.option_context('display.float_format', '{:,.2f}'.format):
		print(df_table.to_string(index=False))

	# Also print PV summary again for convenience
	print('\nPV summary (Alternative A):')
	# PV-corrected investment computed earlier as pv_corrected_investment_total
	pv_inv = float(pv_corrected_investment_total)
	# Compute PV of interruption costs if not available
	if 'pv_cens_A' not in globals():
		years_pv = np.arange(0, horizon_years)
		pv_cens_A = 0.0
		for y in years_pv:
			avg_load = avg_load_year1_mw * ((1 + growth_rate) ** y)
			ENS_per_failure = avg_load * r_perm
			eens = failures_per_year * ENS_per_failure
			cens = eens * 1000.0 * avg_cost_kNOK_per_kWh
			pv_cens_A += cens / ((1 + discount_rate) ** y)

	total_pv_societal_A = pv_inv + float(pv_cens_A)
	print(f'PV-corrected investment (A): {pv_inv:,.2f} NOK')
	print(f'PV of interruption costs (A): {pv_cens_A:,.2f} NOK')
	print(f'Total present-value socio-economic cost (A): {total_pv_societal_A:,.2f} NOK')

except Exception as e:
	print('Error in Task 12:', e)


# %% Task 13: 10-year table and PV socio-economic cost for Alternative B (battery services)
print('\n--- Task 13: 10-year table and PV socio-economic cost for Alternative B ---')
try:
	# Table years 1..10 mapping
	table_years = list(range(1, 11))

	# Use existing reinf_B (year index when B reinforcement happens, or None)
	reinf_B = reinf_B if 'reinf_B' in globals() else (exceed_year_batt if 'exceed_year_batt' in globals() else None)

	# Prepare annual battery operational cost for years 0..9 (we have annual_costs_B from Task 7 for years <10)
	years_task7 = years[years < 10]
	# annual_costs_B exists from Task 7; if not, recompute for years 0..9
	if 'annual_costs_B' not in globals():
		annual_costs_B = []
		for y in years_task7:
			factor = (1 + growth_rate) ** y
			daily_series = (base_aggr * factor)
			hourly_over = (daily_series - P_lim).clip(lower=0)
			hourly_shift = np.minimum(hourly_over, batt_power_mw)
			daily_energy_shift = hourly_shift.sum()
			daily_cost = daily_energy_shift * cost_per_MWh
			annual_cost = daily_cost * representative_days_per_year
			if reinf_B is not None and y >= reinf_B:
				annual_costs_B.append(0.0)
			else:
				annual_costs_B.append(annual_cost)

	# cens_B_by_year exists for years 0..9 from Task 10; if not, recompute similarly
	if 'cens_B_by_year' not in globals() or len(cens_B_by_year) < len(years_task7):
		cens_B_by_year = []
		eens_B_by_year = []
		batt_energy_mwh = 2.0
		batt_power_mw = 1.0
		for y in years_task7:
			avg_load = avg_load_year1_mw * ((1 + growth_rate) ** y)
			ENS_no_batt = avg_load * r_perm
			battery_supply_per_failure = min(batt_power_mw * r_perm, batt_energy_mwh, ENS_no_batt)
			ENS_with_batt = max(0.0, ENS_no_batt - battery_supply_per_failure)
			eens_B = failures_per_year * ENS_with_batt
			cens_B = eens_B * 1000.0 * avg_cost_kNOK_per_kWh
			eens_B_by_year.append(eens_B)
			cens_B_by_year.append(cens_B)

	# Build 10-year table rows
	rows_B = []
	total_nominal = float(investment_new * km_of_new_line)
	for i, year in enumerate(table_years):
		y = year - 1
		inv = corrected_B if (reinf_B is not None and year == (reinf_B + 1)) else 0.0
		op_cost = float(annual_costs_B[y]) if y < len(annual_costs_B) else 0.0
		inter_cost = float(cens_B_by_year[y]) if y < len(cens_B_by_year) else 0.0
		rows_B.append({'Year': year, 'Investment cost (NOK)': inv, 'Operational costs (NOK)': op_cost, 'Interruption costs (NOK)': inter_cost})

	df_table_B = pd.DataFrame(rows_B)
	with pd.option_context('display.float_format', '{:,.2f}'.format):
		print(df_table_B.to_string(index=False))

	# Compute PV components for Alternative B over full horizon (0..horizon_years-1)
	years_pv = np.arange(0, horizon_years)

	# PV of battery operational costs
	pv_batt_opex = 0.0
	for y in years_pv:
		# annual battery cost at year y
		factor = (1 + growth_rate) ** y
		daily_series = (base_aggr * factor)
		hourly_over = (daily_series - P_lim).clip(lower=0)
		hourly_shift = np.minimum(hourly_over, batt_power_mw)
		daily_energy_shift = hourly_shift.sum()
		annual_cost = daily_energy_shift * cost_per_MWh * representative_days_per_year
		# services stop after reinf_B
		if reinf_B is not None and y >= reinf_B:
			annual_cost = 0.0
		pv_batt_opex += annual_cost / ((1 + discount_rate) ** y)

	# PV of interruption costs under B
	pv_cens_B = 0.0
	batt_energy_mwh = 2.0
	batt_power_mw = 1.0
	for y in years_pv:
		avg_load = avg_load_year1_mw * ((1 + growth_rate) ** y)
		ENS_no_batt = avg_load * r_perm
		battery_supply_per_failure = min(batt_power_mw * r_perm, batt_energy_mwh, ENS_no_batt)
		ENS_with_batt = max(0.0, ENS_no_batt - battery_supply_per_failure)
		eens_B = failures_per_year * ENS_with_batt
		cens_B = eens_B * 1000.0 * avg_cost_kNOK_per_kWh
		pv_cens_B += cens_B / ((1 + discount_rate) ** y)

	# PV-corrected investment for B (if reinforcement occurs within horizon)
	corrected_B = 0.0
	if reinf_B is not None and reinf_B < horizon_years:
		t_B = reinf_B
		pv_inv_B = total_nominal / ((1 + discount_rate) ** t_B)
		years_in_service_B = max(0, horizon_years - t_B)
		remaining_life_B = max(0, economic_life_years - years_in_service_B)
		residual_total_B = total_nominal * (remaining_life_B / economic_life_years)
		pv_resid_B = residual_total_B / ((1 + discount_rate) ** horizon_years)
		corrected_B = pv_inv_B - pv_resid_B

	total_pv_B = corrected_B + pv_batt_opex + pv_cens_B

	# Print PV breakdown
	print('\nPV breakdown (Alternative B):')
	print(f'PV-corrected grid investment (B): {corrected_B:,.2f} NOK')
	print(f'PV of battery operational costs (B): {pv_batt_opex:,.2f} NOK')
	print(f'PV of interruption costs (B): {pv_cens_B:,.2f} NOK')
	print('-----------------------------------------')
	print(f'Total present-value socio-economic cost (Alternative B): {total_pv_B:,.2f} NOK')

except Exception as e:
	print('Error in Task 13:', e)


