"""
This file is part of the Python Modular Assessment of Rainfall-Runoff Models
Toolbox (pyMARRMoT).
pyMARRMoT is a free software (GNU GPL v3) and distributed WITHOUT ANY
WARRANTY. See <https://www.gnu.org/licenses/> for details.

Contact: ssheeder@rti.org

This example workflow contains an example application of autocalibration of a
single model to a single catchment. The model utilized in this example is the
Hymod model, which is a conceptual rainfall-runoff model with five stores and
five fluxes. The model is calibrated to a single catchment using the SCE-UA
optimization algorithm implemented in the SPOTPY package. 

This example file calls the 'spotpy_setup' class, which is a wrapper around
the Hymod model and the SPOTPY package. The 'spotpy_setup' class contains
methods to set up the model, define the parameter ranges, and define the
objective function. The 'spotpy_setup' class is used to set up the model for
calibration and to run the SCE-UA optimization algorithm.
"""

import sys, os
print("cwd:", os.getcwd())
print("\n".join(sys.path))

# Ensure the src/ directory is on sys.path so 'pymarrmot' can be imported
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, src_path)

print("Added to path:", src_path)

#print(sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))))

import numpy as np
import spotpy
import spotpy.objectivefunctions as of
#from pymarrmot.functions.autocalibration.spotpy_setup import spotpy_setup as setup
from pymarrmot.functions.autocalibration.spotpy_setup_hymod_hourly import spotpy_setup as setup
import matplotlib.pyplot as plt

#Set the number of model runs
reps = 10000
spotpy_setup = setup(of.kge)
sampler=spotpy.algorithms.sceua(spotpy_setup, dbname='SCEUA_hymod', dbformat='csv')
sampler.sample(reps, ngs=7, kstop=10, peps=0.001, pcento=0.001)
results = spotpy.analyser.load_csv_results('SCEUA_hymod')

#find the run_id with the minimal objective function value
bestindex,bestobjf = spotpy.analyser.get_minlikeindex(results)

#Then we select the best model run:
best_model_run = results[bestindex]

#And filter results for simulation results only:
fields=[word for word in best_model_run.dtype.names if word.startswith('sim')]
best_simulation = list(best_model_run[fields])

#Timeseries plot of the best model run
fig= plt.figure(figsize=(16,9))
ax = plt.subplot(1,1,1)
ax.plot(best_simulation,color='black',linestyle='solid', label='Best objf.='+str(bestobjf))
ax.plot(spotpy_setup.evaluation(),color='red',markersize=3, label='Observation data')
plt.xlabel('Number of Observation Points')
plt.ylabel ('Discharge [cfs]')
plt.legend(loc='upper right')
fig.savefig('SCEUA_hymod_best.png',dpi=300)

#FDC plot of the best model run, exceedance probability on the x-axis and discharge on the y-axis
#Sort the simulated and observed data, and calculate the exceedance probability
sim_sorted = best_simulation
sim_sorted.sort(reverse=True)
obs_sorted = spotpy_setup.evaluation()
obs_sorted.sort(reverse=True)
exceedance_prob = [(x/(len(sim_sorted) + 1)*100) for x in range(len(sim_sorted))]
#get the FDC plot
fig= plt.figure(figsize=(16,9))
ax = plt.subplot(1,1,1)
ax.plot(exceedance_prob, sim_sorted, color='black', linestyle='solid', label='Simulated')
ax.plot(exceedance_prob, obs_sorted, color='red', linestyle='solid', linewidth= 2,label='Observed')
plt.yscale('log')
plt.xlabel('Exceedance Probability')
plt.ylabel('Discharge [cfs]')
plt.legend(loc='upper right')
fig.savefig('SCEUA_hymod_best_FDC.png', dpi=300)
