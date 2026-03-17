from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import rmse

from pymarrmot.models.models.m_29_hymod_5p_5s import m_29_hymod_5p_5s

import os
import pandas as pd
import numpy as np
import importlib
import importlib.util
import sys



class spotpy_setup(object):
    # Following example from https://spotpy.readthedocs.io/en/latest/How_to_link_a_model_to_SPOTPY/
    # Linking a model to SPOTPY is done by following five consecutive steps,
    # which are grouped in a spotpy_setup class

    # Step 1: Define the parameters of the model as class parameters
    
    # parameters that will be calibrated by Spotpy
    # smax = Uniform(low=par_ranges['Smax'][0], high=par_ranges['Smax'][1], optguess=500)    # Smax, Maximum soil moisture storage [mm]
    # b = Uniform(low=par_ranges['b'][0], high=par_ranges['b'][1], optguess=0.1725)     # b, Soil depth distribution parameter [-]
    # a = Uniform(low=par_ranges['a'][0], high=par_ranges['a'][1], optguess=0.8127)      # a, Runoff distribution fraction [-]
    # kf = Uniform(low=par_ranges['kf'][0], high=par_ranges['kf'][1], optguess=0.0404)     # kf, fast flow time parameter [d-1]
    # ks = Uniform(low=par_ranges['ks'][0], high=par_ranges['ks'][1], optguess=0.5592)     # ks, base flow time parameter [d-1]   
    test = 1
    # Step 2: Write the def init function, which takes care of any things which need to be done only once
    def __init__(self, obj_func=None, model_name:str=None, input_climatology:dict=None, trueObs:list=None, input_solver_opts:dict=None):

        # Load the module containing the model
        spec = importlib.util.spec_from_file_location(model_name, 'src\\pymarrmot\\models\\models\\' + model_name + '.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[model_name] = module
        spec.loader.exec_module(module)

        #initialize the model
        model = module.__getattribute__(model_name)()

        #initial storage values - set as average of the range
        # input_s0 = []
        # for key in model.par_ranges:
        #     input_s0.append((model.par_ranges[key][0]+model.par_ranges[key][1])/2) 
        
        #time step
        model.delta_t = 1

        #climate and hydrology data
        model.input_climate = input_climatology
        obs_q = trueObs

        self.obj_func = obj_func
        


    
    # m.input_climate = input_climatology
    
    # m.solver_opts = input_solver_opts
    # m.S0 = np.array(input_s0)
        
  
    # Step 3: Write the def simulation function, which starts your model and returns the results
    def simulation(self,x):
        #update theta with the new parameter set
        self.m.theta = np.array(x)
        
        #Here the model is actualy started with a unique parameter combination that it gets from spotpy for each time the model is called
        (output_ex, output_in, output_ss, output_waterbalance) = self.m.get_output(nargout=4)
        return output_ex['Q'].tolist()
    
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
            # score = self.obj_func(evaluation, simulation)
            # eval_inverse = [0.001 if (i<=0) else 1/i for i in evaluation]
            # sim_inverse = [0.001 if (i<=0) else 1/i for i in simulation]
            # score2 = self.obj_func(eval_inverse, sim_inverse)
            # result = (score + score2)/2
            # like = -1*result

            # Calculation of kge 
            result = self.obj_func(evaluation, simulation)
            like = -1*result


        return like