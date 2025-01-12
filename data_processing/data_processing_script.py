
from whyshift import get_data, degradation_decomp, fetch_model, risk_region
from whyshift.folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime

import numpy as np 
import torch 
import random 
import pickle
import tiktoken
from dataset import *
from preprocess import *
from serialize import *
from embed import *
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None 

# Define states
s = 'AL,AK,AZ,AR,CA,CO,CT,DE,FL,GA,HI,ID,IL,IN,IA,KS,KY,LA,ME,MD,MA,MI,MN,MS,MO,MT,NE,NV,NH,NJ,NM,NY,NC,ND,OH,OK,OR,PA,RI,SC,SD,TN,TX,UT,VT,VA,WA,WV,WI,WY,PR'
all_states = s.split(',')

# Data Processing functions
def data_processing_ACSIncome(task):
    raw_df = get_raw_df(task.task_name, task.state, root_dir = task.root_dir, year=task.year)
    serialize_df = serialize_table(task.data_name, raw_df, target = task.target, group = task.group, root_dir = task.root_dir)
    
    embed_table(task, serialize_df.iloc[:50000], root_dir = task.root_dir)
    print('data processing finished: {}-{}'.format(task.data_name, task.prompt_method))

def data_processing_ACSPubCov(task):
    raw_df = get_raw_df(task.task_name, task.state, root_dir = task.root_dir, year=task.year)
    serialize_df = serialize_table(task.data_name, raw_df, target = task.target, group = task.group, root_dir = task.root_dir)
    embed_table(task, serialize_df.iloc[:50000], root_dir = task.root_dir)
    print('data processing finished: {}-{}'.format(task.data_name, task.prompt_method))

def data_processing_ACSMobility(task):
    raw_df = get_raw_df(task.task_name, task.state, root_dir = task.root_dir, year=task.year)
    serialize_df = serialize_table(task.data_name, raw_df, target = task.target, group = task.group, root_dir = task.root_dir)
    embed_table(task, serialize_df.iloc[:50000], root_dir = task.root_dir)
    print('data processing finished: {}-{}'.format(task.data_name, task.prompt_method))

def data_processing(task_name, prompt_method, state, year = 2018, root_dir = None):
    print(task_name)
    if task_name == 'income':
        task = ACSIncome
        task.task_name = task_name
        task.state  = state
        task.year = year
        task.data_name = 'ACSIncome-{}-{}'.format(state, year)
        task.prompt_method = prompt_method
        task.root_dir = root_dir
        return data_processing_ACSIncome(task)
    elif task_name =='pubcov':
        task = ACSPublicCoverage
        task.task_name = task_name
        task.state = state
        task.year = year
        task.data_name = 'ACSPubCov-{}-{}'.format(state, year)
        task.prompt_method = prompt_method
        task.root_dir = root_dir
        return data_processing_ACSPubCov(task)
    elif task_name == 'mobility':
        task = ACSMobility
        task.task_name = task_name
        task.state = state
        task.year = year
        task.data_name = 'ACSMobility-{}-{}'.format(state, year)
        task.prompt_method = prompt_method
        task.root_dir = root_dir
        return data_processing_ACSMobility(task)

# Processing script
task_name = 'income'
prompt_method_list = ['domainlabel']
root_dir = '/public12_data/fl/shared/share_mala/llm-dro/' + task_name

for prompt_method in prompt_method_list:
    for state in all_states:
        print(state)
        data_processing(task_name = task_name, prompt_method=prompt_method, state = state, year = 2018, root_dir=root_dir)

# Extra information for specific states (Wikipedia information)
extra_info = {
    "CA": "California's economy is the largest of any state within the United States, with a $3.6 trillion gross state product (GSP) as of 2022...",
    "PR": "Puerto Rico is classified as a high income economy by the World Bank and International Monetary Fund...",
    "TX": "As of 2022, Texas had a gross state product (GSP) of $2.4 trillion...",
    "SD": "The current-dollar gross state product of South Dakota was $39.8 billion as of 2010...",
    "NH": "The Bureau of Economic Analysis estimates that New Hampshire's total state product in 2018 was $86 billion..."
}
