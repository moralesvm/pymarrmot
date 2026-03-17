"""
This example workflow contains an example application of calibration of a
single models to a single catchment.
It includes 7 steps:

1. Data preparation
2. Model choice and setup
3. Model solver settings and time-stepping scheme
4. Calibration settings
5. Model calibration
6. Evaluation of calibration results
7. Output visualization

NOTE: this example uses a custom function 'my_cmaes' to perform the
optimization, it is a wrapper around 'cmaes' to ensure inputs and outputs
are consistent with other MATLAB's optimization algorithms (e.g.
'fminsearch' or 'fminsearchbnd').
While the wrapper is provided as part of MARRMoT, it requires the source
code to 'cmaes' to function, it is available at:
http://cma.gforge.inria.fr/cmaes.m
The wrapper is necessary for the optimizer to function within the
MARRMoT_model.calibrate method.
Alternatively any model can be calibrated using any optimization
algorithm using the MARRMoT_model.calc_par_fitness method which returns
the value of an objective function and can be used as input to an
optimizer.
"""

import sys, os
print("cwd:", os.getcwd())
print("\n".join(sys.path))

# Ensure the src/ directory is on sys.path so 'pymarrmot' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pymarrmot.models.models.m_29_hymod_5p_5s import m_29_hymod_5p_5s
import pymarrmot.functions.objective_functions as objective_functions

# 1. Prepare data
df = pd.read_csv('c:/users/ssheeder/repos/pymarrmot/examples/Example_DataSet.csv')

# Create a climatology data input structure
input_climatology = {
    'dates': df['Date'].to_numpy(),      # Daily data: date in ??? format
    'precip': df['Precip'].to_numpy(),   # Daily data: P rate [mm/d]
    'temp': df['Temp'].to_numpy(),       # Daily data: mean T [degree C]
    'pet': df['PET'].to_numpy(),         # Daily data: Ep rate [mm/d]
}



# Extract observed streamflow
Q_obs = df['Q'].to_numpy()

# 2. Define the model settings and create the model object
m = m_29_hymod_5p_5s()  # Create an instance of the model object
par_ranges = m.par_ranges  # Parameter ranges
num_params = m.num_params  # Number of parameters
num_stores = m.num_stores  # Number of stores
input_s0 = np.zeros(num_stores)  # Initial storages (see note in paragraph 5 on model warm-up)

# 3. Define the solver settings
input_solver_opts = {
    'resnorm_tolerance': 0.1,  # Root-finding convergence tolerance;
    # users have reported differences in simulation accuracy (KGE scores) during calibration between Matlab and Octave for a given tolerance.
    # In certain cases, Octave seems to require tighter tolerances to obtain the same KGE scores as Matlab does.
    'resnorm_maxiter': 6  # Maximum number of re-runs
}

# 4. Define calibration settings
# Settings for 'my_cmaes'
# the opts struct is made up of two fields, the names are hardcoded, so
# they cannot be changed:
#    .sigma0:     initial value of sigma
#    .cmaes_opts: struct of options for cmaes, see cmaes documentation
#                 or type cmaes to see list of options and default values

# starting sigma
optim_opts = {'insigma': .3 * (par_ranges[:, 1] - par_ranges[:, 0])}  # starting sigma (this is default, could have left it blank)

# other options
optim_opts['LBounds'] = par_ranges[:, 0]  # lower bounds of parameters
optim_opts['UBounds'] = par_ranges[:, 1]  # upper bounds of parameters
optim_opts['PopSize'] = 4 + np.floor(3 * np.log(num_params))  # population size (default)
optim_opts['TolX'] = 1e-6 * np.min(optim_opts['insigma'])  # stopping criterion on changes to parameters
optim_opts['TolFun'] = 1e-6  # stopping criterion on changes to fitness function
optim_opts['TolHistFun'] = 1e-6  # stopping criterion on changes to fitness function
optim_opts['SaveFilename'] = 'wf_ex_4_cmaesvars.txt'  # output file of cmaes variables
optim_opts['LogFilenamePrefix'] = 'wf_ex_4_'  # prefix for cmaes log-files

# Other useful options
optim_opts['EvalParallel'] = False  # change to true to run in parallel on a pool of CPUs (e.g. on a cluster)
# uncomment to restart r-times until the condition
# r = 2
# optim_opts['Restarts'] = r
# optim_opts['RestartIf'] = 'fmin > -.8' # OF is inverted, so this restarts
#                                        unless the OF (KGE here) > 0.8

# debugging options
#optim_opts['MaxIter'] = 5  # just do 5 iterations, to check if it works
#optim_opts['Seed'] = 1234  # for reproducibility

# initial parameter set
par_ini = np.mean(par_ranges, axis=1)  # same as default value
m.theta = par_ini  # set the initial parameter set

# Choose the objective function
of_name = 'of_kge'  # This function is provided as part of MARRMoT. See ./MARRMoT/Functions/Objective functions
weights = [1, 1, 1]  # Weights for the three KGE components

# Time periods for calibration.
# Indices of timestep to use for calibration, here we are using 1 year for
# warm-up, and 2 years for calibration, leaving the rest of the dataset for
# independent evaluation.
n = len(Q_obs)
warmup = 365
cal_idx = list(range(warmup + 1, warmup + 365 * 2 + 1))
eval_idx = list(range(max(cal_idx), n))

# 5. Calibrate the model
# MARRMoT model objects have a "calibrate" method that takes uses a chosen
# optimization algorithm and objective function to optimize the parameter
# set. See MARRMoT_model class for details.

# first set up the model
m.delta_t = 1
m.input_climate = input_climatology
m.solver_opts = input_solver_opts
m.S0 = input_s0

##Debug - Remove
par_ini = np.array([2.17378236e+02, 2.66901321e-03, 1.24947658e-01, 9.86336592e-01, 9.78946001e-01])

par_opt, of_cal, stopflag, output = m.calibrate(
    Q_obs,  # observed streamflow
    cal_idx,  # timesteps to use for model calibration
    'my_cmaes',  # function to use for optimization (must have same structure as fminsearch)
    par_ini,  # initial parameter estimates
    optim_opts,  # options to optim_fun
    of_name,  # name of objective function to use
    True, True,  # should the OF be inversed?   Should I display details about the calibration?
    weights  # additional arguments to of_name
)

# 6. Evaluate the calibrated parameters on unseen data
# Run the model with calibrated parameters, get only the streamflow
Q_sim = m.get_streamflow(par_opt)

# Compute evaluation performance
obj_func = getattr(objective_functions, of_name) # Objective function (here 'of_KGE')
of_eval = obj_func(Q_obs,       # Observed flow during evaluation period
                   Q_sim,       # Simulated flow during evaluation period, using calibrated parameters
                   eval_idx,    # Indices of evaluation period
                   weights)     # KGE component weights
of_eval = of_eval[0]  # Extract the KGE value from the tuple

# 7. Visualize the results
# Prepare a time vector
t = [datetime.fromtimestamp(t) for t in m.input_climate['dates']]

# Compare simulated and observed streamflow

plt.figure(figsize=(10, 6), facecolor='w')
plt.box(True)

# Flows
plt.plot(t, Q_obs, 'k', label='Q_obs')
plt.plot(t, Q_sim, 'r', label='Q_sim')

# Dividing line
plt.axvline(t[max(cal_idx)], color='b', linestyle='--', linewidth=2, label='Cal // Eval')
plt.axvline(t[warmup], color='g', linestyle='--', linewidth=2, label='warmup // Cal')

# Legend & text
plt.legend(loc='upper left')
plt.title('Model calibration and evaluation results')
plt.ylabel('Streamflow [mm/d]')
plt.xlabel('Time [d]')

max_obs = np.max(Q_obs)
max_sim = np.max(Q_sim)
max_val = 1.05 * max(max_obs, max_sim)
max_t = max(t)
min_t = min(t)
del_t = max(t) - min(t)

plt_txt = 'Cal ' + of_name + ' = %.2f' % round(of_cal,2)
plt_txt += '\nEval ' + of_name + ' = %.2f' % round(of_eval,2)

plt.text(min_t + del_t *.85, max_val * .85, plt_txt, fontsize = 10, bbox=dict(facecolor='white', edgecolor='red'))
plt.xticks(rotation=45)

plt.ylim([0, max_val])
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show(block=True)