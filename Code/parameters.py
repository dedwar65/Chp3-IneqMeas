'''
This file specifies parameters for the agent types and empirical targets.
'''
import numpy as np
import csv
import matplotlib.pyplot as plt
import yaml
from copy import deepcopy
from HARK.core import AgentPopulation
from HARK.distribution import Uniform, Lognormal
import pandas as pd
from HARK.Calibration.Income.IncomeTools import (
    CGM_income,
    Cagetti_income,
    parse_income_spec,
    parse_time_params,
)
from HARK.datasets.life_tables.us_ssa.SSATools import parse_ssa_life_table
from HARK.datasets.SCF.WealthIncomeDist.SCFDistTools import income_wealth_dists_from_scf
from utilities import get_lorenz_shares, calcEmpMoments, AltIndShockConsumerType
import os

MyAgentType = AltIndShockConsumerType

script_dir = os.path.dirname(os.path.abspath(__file__))
data_location = os.path.join(script_dir, '../Data/')
specs_location = os.path.join(script_dir, '../Specifications/')
SpecificationFilename = 'LCbetaDistNetWorth.yaml'

with open(specs_location + SpecificationFilename, 'r') as f:
    spec_raw = f.read()
    f.close()
yaml_params = yaml.safe_load(spec_raw)
print('Loading a specification called ' + yaml_params['description'])

tag = yaml_params['tag']
model = yaml_params["model"]

# Choose basic specification parameters
HetParam = yaml_params['HetParam']
DstnTypeName = yaml_params['DstnType']
HetTypeCount = yaml_params['HetTypeCount']
TotalAgentCount = HetTypeCount*yaml_params['AgentsPerType']
LifeCycle = yaml_params['LifeCycle']

# Specify search parameters
center_range = yaml_params['center_range']
spread_range = yaml_params['spread_range']

# Setup basics for computing empirical targets from the SCF
TargetPercentiles = yaml_params['TargetPercentiles']
wealth_data_file = yaml_params['wealth_data_file']
wealth_col = yaml_params['wealth_col']
weight_col = yaml_params['weight_col']
income_col = yaml_params['income_col']

# Import the wealth and income data to be matched in estimation
f = open(data_location + "/" + wealth_data_file)
wealth_data_reader = csv.reader(f, delimiter="\t")
wealth_data_raw = list(wealth_data_reader)
wealth_data = np.zeros(len(wealth_data_raw)) + np.nan
weights_data = deepcopy(wealth_data)
income_data = deepcopy(wealth_data)
for j in range(len(wealth_data_raw)):
    # skip the row of headers
    wealth_data[j] = float(wealth_data_raw[j][wealth_col])
    weights_data[j] = float(wealth_data_raw[j][weight_col])
    income_data[j] = float(wealth_data_raw[j][income_col])

# Calculate empirical moments to be used as targets
empirical_moments = calcEmpMoments(wealth_data, income_data, weights_data, TargetPercentiles)
emp_KY_ratio = empirical_moments[0]
emp_lorenz = empirical_moments[1]

# Define a mapping from (center,spread) to the actual parameters of the distribution.
# For each class of distributions you want to allow, there needs to be an entry for
# DstnParam mapping that says what (center,spread) represents for that distribution.
if DstnTypeName == 'Uniform':
    DstnType = Uniform
    DstnParamMapping = lambda center, spread : [center-spread, center+spread]
elif DstnTypeName == 'Lognormal':
    DstnType = Lognormal
    DstnParamMapping = lambda center, spread : [np.log(center) - 0.5 * spread**2, spread]
else:
    print('Oh no! You picked an invalid distribution type!')

# Define a baseline parameter dictionary; this content will be in a YAML file later
base_param_filename = yaml_params['base_param_filename']
with open(specs_location + base_param_filename + '.yaml', 'r') as f:
    init_raw = f.read()
    f.close()
BaseParamDict = {
    "BaseAgentCount" : TotalAgentCount,
    "track_vars": ['aLvl','pLvl','WeightFac']
}
BaseParamDict.update(yaml.safe_load(init_raw)) # Later, add conditions to include other agent types

# Adjust survival probabilities from SSA tables using education cohort adjustments; 
# method provided by Brown, Liebman, and Pollett (2002).
mort_data_file = yaml_params['mort_data_file']

birth_age = BaseParamDict['birth_age']
death_age = BaseParamDict['death_age']

# Compute base mortality rates for the specified age range
base_liv_prb = parse_ssa_life_table(
        female=True, cross_sec=True, year=2004, min_age=birth_age, max_age=death_age - 1
    )

# Import adjustments for education and apply them to the base mortality rates
f = open(data_location + "/" + mort_data_file)
adjustment_reader = csv.reader(f, delimiter=" ")
raw_adjustments = list(adjustment_reader)
nohs_death_probs = []
hs_death_probs = []
c_death_probs = []
for j in range(death_age - birth_age):
    if j < 76:
        nohs_death_probs += [base_liv_prb[j] * float(raw_adjustments[j][1])]
        hs_death_probs += [base_liv_prb[j] * float(raw_adjustments[j][2])]
        c_death_probs += [base_liv_prb[j] * float(raw_adjustments[j][3])]
    else:
        nohs_death_probs += [base_liv_prb[j] * float(raw_adjustments[75][1])]
        hs_death_probs += [base_liv_prb[j] * float(raw_adjustments[75][2])]
        c_death_probs += [base_liv_prb[j] * float(raw_adjustments[75][3])]

# Here define the population of agents for the simulation
if LifeCycle:
    adjust_infl_to = 2004
    income_calib = Cagetti_income

    # Define fractions of education types
    nohs_frac = 0.11
    hs_frac = 0.54
    college_frac = 0.35
    
    # Define dictionaries for life cycle version of the model. Should also be in Yaml file
    # Note: missing survival probabilites conditional on education level.
    nohs_dict = deepcopy(BaseParamDict)
    income_params = parse_income_spec(
        age_min=birth_age,
        age_max=death_age,
        adjust_infl_to=adjust_infl_to,
        **income_calib["NoHS"],
        SabelhausSong=True,
    )
    dist_params = income_wealth_dists_from_scf(
        base_year=adjust_infl_to, age=birth_age, education="NoHS", wave=1995
    )
    time_params = parse_time_params(age_birth=birth_age, age_death=death_age)
    nohs_dict.update(time_params)
    nohs_dict.update(dist_params)
    nohs_dict.update(income_params)
    nohs_dict.update({"LivPrb": nohs_death_probs})
    nohs_dict['BaseAgentCount'] = TotalAgentCount*nohs_frac
    
    hs_dict = deepcopy(BaseParamDict)
    income_params = parse_income_spec(
        age_min=birth_age,
        age_max=death_age,
        adjust_infl_to=adjust_infl_to,
        **income_calib["HS"],
        SabelhausSong=True,
    )
    dist_params = income_wealth_dists_from_scf(
        base_year=adjust_infl_to, age=birth_age, education="HS", wave=1995
    )
    time_params = parse_time_params(age_birth=birth_age, age_death=death_age)
    hs_dict.update(time_params)
    hs_dict.update(dist_params)
    hs_dict.update(income_params)
    hs_dict.update({"LivPrb": hs_death_probs})
    hs_dict['BaseAgentCount'] = TotalAgentCount*hs_frac
    
    college_dict = deepcopy(BaseParamDict)
    income_params = parse_income_spec(
        age_min=birth_age,
        age_max=death_age,
        adjust_infl_to=adjust_infl_to,
        **income_calib["College"],
        SabelhausSong=True,
    )
    dist_params = income_wealth_dists_from_scf(
        base_year=adjust_infl_to, age=birth_age, education="College", wave=1995
    )
    time_params = parse_time_params(age_birth=birth_age, age_death=death_age)
    college_dict.update(time_params)
    college_dict.update(dist_params)
    college_dict.update(income_params)
    college_dict.update({"LivPrb": c_death_probs})
    college_dict['BaseAgentCount'] = TotalAgentCount*college_frac
    
    # Make base agent types
    DropoutType = MyAgentType(**nohs_dict)
    HighschType = MyAgentType(**hs_dict)
    CollegeType = MyAgentType(**college_dict)
    BaseTypeCount = 3
    BasePopulation = [DropoutType, HighschType, CollegeType]

else:
    IHbaseType = MyAgentType(**BaseParamDict)
    BaseTypeCount = 1
    BasePopulation = [IHbaseType]

# Set the agent population
MyPopulation = []
for n in range(HetTypeCount):
    MyPopulation += deepcopy(BasePopulation)
    
# Store optimal parameters here
opt_center = None
opt_spread = None
lorenz_distance = None

