from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

from pymarrmot.models.models.m_33_sacramento_11p_5s import m_33_sacramento_11p_5s

import pandas as pd
import numpy as np

class spotpy_setup_sacramento(object):
    # Following example from https://spotpy.readthedocs.io/en/latest/How_to_link_a_model_to_SPOTPY/
    # Linking a model to SPOTPY is done by following five consecutive steps,
    # which are grouped in a spotpy_setup class

    # Step 1: Define the parameters of the model as class parameters

    par_ranges = np.array([[0       , 1],         # pctim, Fraction impervious area [-]
                            [1       , 2000],     # smax, Maximum total storage depth [mm]
                            [0.005   , 0.995],    # f1, fraction of smax that is Maximum upper zone tension water storage [mm]
                            [0.005   , 0.995],    # f2, fraction of smax-S1max that is Maximum upper zone free water storage [mm]
                            [0       , 1],        # kuz, Interflow runoff coefficient [d-1]
                            [0       , 7],        # rexp, Base percolation rate non-linearity factor [-]
                            [0.005   , 0.995],    # f3, fraction of smax-S1max-S2max that is  Maximum lower zone tension water storage [mm]
                            [0.005   , 0.995],    # f4, fraction of smax-S1max-S2max-S3max that is  Maximum lower zone primary free water storage [mm]
                            [0       , 1],        # pfree, Fraction of percolation directed to free water stores [-]
                            [0       , 1],        # klzp, Primary baseflow runoff coefficient [d-1]
                            [0       , 1]])       # klzs, Supplemental baseflow runoff coefficient [d-1]
    
    # parameters that will be calibrated by Spotpy
    pctim = Uniform(low=par_ranges[0][0], high=par_ranges[0][1], optguess=0.1)        # pctim, Fraction impervious area [-]
    smax = Uniform(low=par_ranges[1][0], high=par_ranges[1][1], optguess=500)      # smax, Maximum total storage depth [mm]
    f1 = Uniform(low=par_ranges[2][0], high=par_ranges[2][1], optguess=0.2)        # f1, fraction of smax that is Maximum upper zone tension water storage (uztwm) [-]
    f2 = Uniform(low=par_ranges[3][0], high=par_ranges[3][1], optguess=0.5)        # f2, fraction of smax-uztwm that is Maximum upper zone free water storage (uzfwm) [-]
    kuz = Uniform(low=par_ranges[4][0], high=par_ranges[4][1], optguess=0.1)      # kuz, Interflow runoff coefficient [d-1]
    rexp = Uniform(low=par_ranges[5][0], high=par_ranges[5][1], optguess=1)        # rexp, Base percolation rate non-linearity factor [-]
    f3 = Uniform(low=par_ranges[6][0], high=par_ranges[6][1], optguess=0.3)        # f3, fraction of smax-uztwm-uzfwm that is Maximum lower zone tension water storage (lztwm) [-]
    f4 = Uniform(low=par_ranges[7][0], high=par_ranges[7][1], optguess=0.5)        # f4, fraction of smax-uztwm-uzfwm-lztwm that is Maximum lower zone primary free water storage (lzfwpm) [-]
    pfree = Uniform(low=par_ranges[8][0], high=par_ranges[8][1], optguess=0.5)    # pfree, Fraction of percolation directed to free water stores [-]
    klzp = Uniform(low=par_ranges[9][0], high=par_ranges[9][1], optguess=0.5)      # klzp, Primary baseflow runoff coefficient [d-1]
    klzs = Uniform(low=par_ranges[10][0], high=par_ranges[10][1], optguess=0.5)    # klzs, Supplemental baseflow runoff coefficient [d-1]
    
    # Step 2: Write the def init function, which takes care of any things which need to be done only once
    def __init__(self,obj_func=None):
        self.obj_func = obj_func
        
    #initial storage values - set as average of the range
    input_s0 = []
    #loop over par_ranges dictionary and get average value for each key
    for key in par_ranges:
        input_s0.append((par_ranges[key][0]+par_ranges[key][1])/2) 

    #Define the solver settings
    input_solver_opts = {
        'resnorm_tolerance': 0.1,
        'resnorm_maxiter': 6
    }

    #Create a model object
    m = m_33_sacramento_11p_5s()
    m.delta_t = 1/24  # hourly time step

    df = pd.read_csv('C:/Users/ssheeder/Repos/pymarrmot/model_tests_hourly/forcing/forcing_daily_04288000.csv')
    # Create a climatology data input structure
    input_climatology = {
        'dates': df['value_time'].to_numpy(),      # Daily data: date in 'm/d/yyyy' format
        'precip': df['ppt_mm'].to_numpy(),   # Daily data: P rate [mm/d]
        'temp': df['temp_c'].to_numpy(),       # Daily data: mean T [degree C]
        'pet': df['pet_mm'].to_numpy(),         # Daily data: Ep rate [mm/d]
    }
    trueObs = df['obs_mm/day'].to_list() # Daily data: Q rate [cms]

    # Clean up the evaluation and simulation data - remove any NaNs and missing values from evaluation dataset, and corresponding values from simulation dataset
    nan_indices = np.argwhere(np.isnan(trueObs))
    #missing_indices = np.argwhere(trueObs == "")
    #unique_values = list(set(nan_indices + missing_indices))
    if len(nan_indices) > 0:
        print(f"WARNING: {len(nan_indices)} NaN and missing values found in evaluation dataset. These values will be removed from the evaluation and simulation datasets before calculating model fitness.")
        trueObs = np.delete(trueObs, nan_indices)

    m.input_climate = input_climatology
    
    m.solver_opts = input_solver_opts
    m.S0 = np.array(input_s0)
        
    # Step 3: Write the def simulation function, which starts your model and returns the results
    def simulation(self,x):
        #update theta with the new parameter set
        self.m.theta = np.array(x)
        
        #Here the model is actualy started with a unique parameter combination that it gets from spotpy for each time the model is called
        (output_ex, output_in, output_ss, output_waterbalance) = self.m.get_output(nargout=4)
        sim_q = output_ex['Q']
        if self.nan_indices.size > 0:
            sim_q = np.delete(sim_q, self.nan_indices)
        return sim_q.tolist()
    
    # Step 4: Write the def evaluation function, which returns the observations
    def evaluation(self):
        return self.trueObs
    
    # Step 5: Write the def objectivefunction function, which returns the objective function value
    def objectivefunction(self, simulation, evaluation, params=None):
    #SPOTPY expects to get one or multiple values back, 
    #that define the performance of the model run
       
        if not self.obj_func:
            # This is used if not overwritten by user
            score = rmse(evaluation, simulation)
            like = score
        else:
            # Spotpy minimizes the objective function, so for objective functions where fitness improves with increasing result, we need to multiply by -1
            
            # calculation of kge-lowflow (based on kge being selected as objective function)
            score = self.obj_func(evaluation, simulation)
            eval_inverse = [1000 if (i<=0) else 1/i for i in evaluation]
            sim_inverse = [1000 if (i<=0) else 1/i for i in simulation]
            score2 = self.obj_func(eval_inverse, sim_inverse)
            result = (score + score2)/2
            like = -1*result
        return like