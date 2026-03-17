from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

from pymarrmot.models.models.m_33_sacramento_11p_5s import m_33_sacramento_11p_5s

import pandas as pd
import numpy as np

class spotpy_setup(object):
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

    # class variables
    m = m_33_sacramento_11p_5s()
    m.delta_t = 1/24  # time step in days (1 hour)
    trueObs = None

    # Step 2: Write the def init function, which takes care of any things which need to be done only once
    def __init__(self,obj_func=None):
        self.obj_func = obj_func
        
        #initial storage values - set as average of the range
        input_s0 = []
        #loop over par_ranges dictionary and get average value for each key
        for key in self.par_ranges:
            input_s0.append((self.par_ranges[key][0]+self.par_ranges[key][1])/2) 

        #Define the solver settings
        input_solver_opts = {
            'resnorm_tolerance': 0.1,
            'resnorm_maxiter': 6
        }

        self.m.S0 = np.array(input_s0)
        self.m.solver_opts = input_solver_opts
        
        #USGS 03463300 - SOUTH TOE RIVER NEAR CELO, NC - usgsbasin-03463300_combined.parquet
        #USGS 02138500 - LINVILLE RIVER NEAR NEBO, NC - usgsbasin-02138500_combined.parquet
        #USGS 03441000 - DAVIDSON RIVER NEAR BREVARD, NC - usgsbasin-03441000_combined.parquet

        df = pd.read_parquet('C:/Users/ssheeder/Repos/pymarrmot/forcing/pymarrmot/combined_forcing/usgsbasin-03463300_combined.parquet')

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
       #left join on 'value_time' to align simulation and evaluation data
        merged_df = pd.merge(evaluation_df, simulation_df, on='value_time', how='left')
        evaluation_array = merged_df['discharge_mm'].to_numpy()
        simulation_array = merged_df['simulated_discharge_mm'].to_numpy()

        if not self.obj_func:
            # This is used if not overwritten by user
            score = rmse(evaluation_array, simulation_array)
            like = score
        else:
            # Spotpy minimizes the objective function, so for objective functions where fitness improves with increasing result, we need to multiply by -1
            
            # calculation of kge-lowflow (based on kge being selected as objective function)
            score = self.obj_func(evaluation_array, simulation_array)
            eval_inverse = [1000 if (i<=0) else 1/i for i in evaluation]
            sim_inverse = [1000 if (i<=0) else 1/i for i in simulation]
            score2 = self.obj_func(eval_inverse, sim_inverse)
            result = (score + score2)/2
            like = -1*result
        return like