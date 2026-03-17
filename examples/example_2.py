"""
This example workflow contains an example application of a single model
to a single catchment.
It includes 5 steps:

1. Data preparation
2. Model choice and setup
3. Model solver settings
4. Model generation and set-up
5. Model runs
6. Output visualization
"""

import sys, os
print("cwd:", os.getcwd())
print("\n".join(sys.path))
 
# Ensure the src/ directory is on sys.path so 'pymarrmot' can be imported
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, src_path)

print("Added to path:", src_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymarrmot.models.models.m_29_hymod_5p_5s import m_29_hymod_5p_5s

# 1. Prepare data
#df = pd.read_csv('c:/users/ssheeder/repos/pymarrmot/examples/Example_DataSet.csv')
example_path = os.path.join(os.path.dirname(__file__), "Example_DataSet.csv")
df = pd.read_csv(example_path)

# Create a climatology data input structure
input_climatology = {
    'dates': df['Date'].to_numpy(),      # Daily data: date in 'm/d/yyyy' format
    'precip': df['Precip'].to_numpy(),   # Daily data: P rate [mm/d]
    'temp': df['Temp'].to_numpy(),       # Daily data: mean T [degree C]
    'pet': df['PET'].to_numpy(),         # Daily data: Ep rate [mm/d]
}

# 2. Define the model settings
input_s0 = np.array([1, 7, 3, 8, 22])

# 3. Define the solver settings
input_solver_opts = {
    'resnorm_tolerance': 0.1,
    'resnorm_maxiter': 6
}

# 4. Create a model object
m = m_29_hymod_5p_5s()
m.delta_t = 1
m.input_climate = input_climatology
m.solver_opts = input_solver_opts
m.S0 = input_s0

model_range = m.par_ranges

# 5. Run the model and extract all outputs (10 runs, random parameterization)
numSample = 10
num_par = m.num_params

results_mc_sampling = []

results_mc_sampling.append(['parameter_values', 'output_ex', 'output_in', 'output_ss', 'output_wb'])

for i in range(numSample):
    #input_theta = model_range[:, 0] + np.random.rand(num_par) * (model_range[:, 1] - model_range[:, 0])
    m.theta = model_range[:, 0] + np.random.rand(num_par) * (model_range[:, 1] - model_range[:, 0])

    output_ex, output_in, output_ss, output_waterbalance = m.get_output(nargout=4) #(4, [], [], m.theta)

    results_mc_sampling.append([m.theta, output_ex, output_in, output_ss, output_waterbalance])

# 5. Analyze the outputs
t = input_climatology['dates']
streamflow_observed = df['Q'].to_numpy()

plt.figure(figsize=(10, 6))
for i in range(1, numSample + 1):
    plt.plot(t, results_mc_sampling[i][1]['Q'], color=[0.5, 0.5, 0.5])

plt.plot(t, streamflow_observed, 'r:')
plt.legend(['Simulated'] * numSample + ['Observed'])
plt.title('Monte Carlo sampling results')
plt.ylabel('Streamflow [mm/d]')
plt.xlabel('Time [d]')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()