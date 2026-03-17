from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

from pymarrmot.models.models.m_29_hymod_5p_5s import m_29_hymod_5p_5s

import pandas as pd
import numpy as np

class spotpy_setup(object):
    # Following example from https://spotpy.readthedocs.io/en/latest/How_to_link_a_model_to_SPOTPY/
    # Linking a model to SPOTPY is done by following five consecutive steps,
    # which are grouped in a spotpy_setup class

    # Step 1: Define the parameters of the model as class parameters
    par_ranges = {
        'Smax': [1, 2000],   
        'b': [0.01, 10],        
        'a': [0.01, 1],         
        'kf': [0.01, 1],       
        'ks': [0.01, 1]        
        }
    
    # parameters that will be calibrated by Spotpy
    smax = Uniform(low=par_ranges['Smax'][0], high=par_ranges['Smax'][1], optguess=500)    # Smax, Maximum soil moisture storage [mm]
    b = Uniform(low=par_ranges['b'][0], high=par_ranges['b'][1], optguess=0.1725)          # b, Soil depth distribution parameter [-]
    a = Uniform(low=par_ranges['a'][0], high=par_ranges['a'][1], optguess=0.8127)          # a, Runoff distribution fraction [-]
    kf = Uniform(low=par_ranges['kf'][0], high=par_ranges['kf'][1], optguess=0.0404)       # kf, fast flow time parameter [d-1]
    ks = Uniform(low=par_ranges['ks'][0], high=par_ranges['ks'][1], optguess=0.5592)       # ks, base flow time parameter [d-1]   
    
    # class variables
    m = m_29_hymod_5p_5s()
    m.delta_t = 1/24  # time step in days (1 hour)
    trueObs = None

    # Step 2: Write the def init function, which takes care of any things which need to be done only once
    def __init__(self,obj_func=None):
        self.obj_func = obj_func
        
        # Initial storage values
        input_s0 = np.array([15,  # Initial soil moisture storage [mm]
                     7,           # Initial fast flow 1 storage [mm]
                     3,           # Initial fast flow 2 storage [mm]
                     8,           # Initial fast flow 3 storage [mm]
                     22])         # Initial slow flow storage [mm]

        #Define the solver settings
        input_solver_opts = {
            'resnorm_tolerance': 0.1,
            'resnorm_maxiter': 6
        }

        self.m.S0 = input_s0
        self.m.solver_opts = input_solver_opts
        
        #USGS 03463300 - SOUTH TOE RIVER NEAR CELO, NC - usgsbasin-03463300_combined.parquet
        #USGS 02138500 - LINVILLE RIVER NEAR NEBO, NC - usgsbasin-02138500_combined.parquet
        #USGS 03441000 - DAVIDSON RIVER NEAR BREVARD, NC - usgsbasin-03441000_combined.parquet

        df = pd.read_parquet('C:/Users/ssheeder/Repos/pymarrmot/forcing/pymarrmot/combined_forcing/12_year/usgsbasin-03463300_combined.parquet')

        # Create a climatology data input structure
        input_climatology = {
            'dates': df['value_time'].to_numpy(),  
            'precip': df['precip_mm'].to_numpy(),     
            'temp': df['temp_c'].to_numpy(),       
            'pet': df['pet_mm'].to_numpy()      
        }
        self.trueObs = df['discharge_mm'].tolist()
        self.m.input_climate = input_climatology
            
    # Step 3: Write the def simulation function, which starts your model and returns the results
    def simulation(self,x):
        #update theta with the new parameter set
        self.m.theta = np.array(x)
        
        #Here the model is actually started with a unique parameter combination that it gets from spotpy for each time the model is called
        (output_ex, output_in, output_ss, output_waterbalance) = self.m.get_output(nargout=4)
        return output_ex['Q'].tolist()
    
    # Step 4: Write the def evaluation function, which returns the observations
    def evaluation(self):
        return self.trueObs
    
    # Step 5: Write the def objectivefunction function, which returns the objective function value
    def objectivefunction(self, simulation, evaluation, params=None):
    #SPOTPY expects to get one or multiple values back, 
    #that define the performance of the model run
       #add simulation and evaluation lists to dataframe
        simulation_df = pd.DataFrame({'value_time': self.m.input_climate['dates'], 'simulated_discharge_mm': simulation})
        evaluation_df = pd.DataFrame({'value_time': self.m.input_climate['dates'], 'discharge_mm': evaluation})
       #inner join on 'value_time' to align simulation and evaluation data
        merged_df = pd.merge(evaluation_df, simulation_df, on='value_time', how='inner')
        #elimination of any rows with NaN values
        merged_df = merged_df.dropna(subset=['discharge_mm', 'simulated_discharge_mm'])
        evaluation_list = merged_df['discharge_mm'].tolist()
        simulation_list = merged_df['simulated_discharge_mm'].tolist()

        if not self.obj_func:
            # This is used if not overwritten by user
            score = rmse(evaluation_list, simulation_list)
            like = score
        else:
            # Spotpy minimizes the objective function, so for objective functions where fitness improves with increasing result, we need to multiply by -1
            
            # calculation of kge-lowflow (based on kge being selected as objective function)
            score = self.obj_func(evaluation_list, simulation_list)
            # removing the low flow component to focus on high flows for the FFF project
            # eval_inverse = [1000 if (i<=0) else 1/i for i in evaluation_list]
            # sim_inverse = [1000 if (i<=0) else 1/i for i in simulation_list]
            # score2 = self.obj_func(eval_inverse, sim_inverse)
            # result = (score + score2)/2
            like = -1*score
        return like