import os
import pickle 
from tqdm import tqdm
import sys
sys.path.append("/public12_data/fl/LLM-Tabular-Shifts-main/data_processing")


from embed import *
from train import *

from src.mlp_concat import *
from src.mlp_e5 import *
from src.mlp import *

from utils import (
    get_raw_data, 
    sample_config, 
    fetch_model, 
    sample_data, 
    get_e5_data,
    sample_val_data
)

s = 'AL,AK,AZ,AR,CA,CO,CT,DE,FL,GA,HI,ID,IL,IN,IA,KS,KY,LA,ME,MD,MA,MI,MN,MS,MO,MT,NE,NV,NH,NJ,NM,NY,NC,ND,OH,OK,OR,PA,RI,SC,SD,TN,TX,UT,VT,VA,WA,WV,WI,WY,PR'
all_states = s.split(',')
state_to_idx = {state: idx for idx, state in enumerate(all_states)}

state_dict = {
    'AL': 'Alabama','AK': 'Alaska','AZ': 'Arizona','AR': 'Arkansas','CA': 'California','CO': 'Colorado','CT': 'Connecticut',
    'DE': 'Delaware','FL': 'Florida','GA': 'Georgia','HI': 'Hawaii','ID': 'Idaho','IL': 'Illinois','IN': 'Indiana','IA': 'Iowa',
    'KS': 'Kansas','KY': 'Kentucky','LA': 'Louisiana','ME': 'Maine','MD': 'Maryland','MA': 'Massachusetts','MI': 'Michigan',
    'MN': 'Minnesota','MS': 'Mississippi','MO': 'Missouri','MT': 'Montana','NE': 'Nebraska','NV': 'Nevada','NH': 'New Hampshire',
    'NJ': 'New Jersey','NM': 'New Mexico','NY': 'New York','NC': 'North Carolina','ND': 'North Dakota','OH': 'Ohio','OK': 'Oklahoma',
    'OR': 'Oregon','PA': 'Pennsylvania','RI': 'Rhode Island','SC': 'South Carolina','SD': 'South Dakota','TN': 'Tennessee','TX': 'Texas',
    'UT': 'Utah','VT': 'Vermont','VA': 'Virginia','WA': 'Washington','WV': 'West Virginia','WI': 'Wisconsin','WY': 'Wyoming',
    'PR': 'Puerto Rico'
}
source_state = 'HI'
source_state_list = source_state.split(' ')
def load_data(method, state, root_dir, num_train=20000, num_val=500, num_test=5000):
    if method == 'concat':
        X, y = get_concat_data('income', state, root_dir)
    elif method == 'e5':
        X, y = get_e5_data('income', 'domainlabel', state, root_dir)
    elif method == 'one_hot':
        X, y = get_onehot_data('income', state, False, root_dir, year=2018)

    n = X.shape[0]
    if n < num_train + num_val + num_test:
        num_train = int(num_train * n / (num_train + num_val + num_test))
        num_val = int(num_val * n / (num_train + num_val + num_test))
        num_test = n - num_train - num_val

    # Setting the random seed to ensure reproducibility
    np.random.seed(42)  # You can use any number here as your seed
    # Combining the data into a single array for shuffling
    data = np.column_stack((X, y))
    np.random.shuffle(data)
    # Splitting the data back into features and labels
    X, y = data[:, :-1], data[:, -1]
    # Splitting the data into train, validation, and test sets
    trainx, trainy = X[:num_train], y[:num_train]
    valx, valy = X[num_train:num_train+num_val], y[num_train:num_train+num_val]
    testx, testy = X[-num_test:], y[-num_test:]
    return trainx, trainy, valx, valy, testx, testy
for state in source_state_list:
    root_dir = '/public12_data/fl/shared/share_mala/llm-dro/income/'
    trainx, trainy, valx, valy, testx, testy = load_data('e5', state, root_dir)
    # concat all train data into one
    if state == source_state_list[0]:
        X_train = trainx
        y_train = trainy
    else:
        X_train = np.concatenate((X_train, trainx), axis=0)
        y_train = np.concatenate((y_train, trainy), axis=0)
save_dir = '/public12_data/fl/shared/share_mala/llm-dro/'
task_name = 'income'
source_state_str = 'CA'
embedding_method = 'e5'
prompt_method = 'domainlabel'
model_dir = f'{save_dir}/save_models/{task_name}/{source_state_str}/{embedding_method}/{prompt_method}/'
domainlabel_model = MLPe5Classifier(input_dim=X_train.shape[1], num_classes = 2, hidden_dim=64)
y_train = torch.tensor(y_train).long()
domainlabel_model.fit(X_train, y_train)

domainlabel_source_train_acc_dict, domainlabel_source_val_acc_dict, domainlabel_source_test_acc_dict = dict(), dict(), dict()
domainlabel_source_train_f1_dict, domainlabel_source_val_f1_dict, domainlabel_source_test_f1_dict = dict(), dict(), dict()

## report training, val, testing performance for each state
for state in source_state_list:
    root_dir = '/public12_data/fl/shared/share_mala/llm-dro/income/'
    trainx, trainy, valx, valy, testx, testy = load_data('e5', state, root_dir)
    train_acc, train_f1 = domainlabel_model.score(trainx, trainy)
    val_acc, val_f1 = domainlabel_model.score(valx, valy)
    test_acc, test_f1 = domainlabel_model.score(testx, testy)
    
    domainlabel_source_train_acc_dict[state], domainlabel_source_val_acc_dict[state], domainlabel_source_test_acc_dict[state] = train_acc, val_acc, test_acc
    domainlabel_source_train_f1_dict[state], domainlabel_source_val_f1_dict[state], domainlabel_source_test_f1_dict[state] = train_f1, val_f1, test_f1

print("Source State: ")
print(f"average train acc: {np.mean(list(domainlabel_source_train_acc_dict.values())):.3f}, average train f1: {np.mean(list(domainlabel_source_train_f1_dict.values())):.3f}")
print(f"average val acc: {np.mean(list(domainlabel_source_val_acc_dict.values())):.3f}, average val f1: {np.mean(list(domainlabel_source_val_f1_dict.values())):.3f}")
print(f"average test acc: {np.mean(list(domainlabel_source_test_acc_dict.values())):.3f}, average test f1: {np.mean(list(domainlabel_source_test_f1_dict.values())):.3f}")
domainlabel_target_val_acc_dict, domainlabel_target_test_acc_dict =  dict(), dict()
domainlabel_target_val_f1_dict, domainlabel_target_test_f1_dict =  dict(), dict()
for state in ['IA']:
    if state not in source_state_list:
        root_dir = '/public12_data/fl/shared/share_mala/llm-dro/income/'
        trainx, trainy, valx, valy, testx, testy = load_data('e5', state, root_dir)
        val_acc, val_f1 = domainlabel_model.score(valx, valy)
        test_acc, test_f1 = domainlabel_model.score(testx, testy)
        
        domainlabel_target_val_acc_dict[state], domainlabel_target_test_acc_dict[state] = val_acc, test_acc
        domainlabel_target_val_f1_dict[state], domainlabel_target_test_f1_dict[state] = val_f1, test_f1

print("Target State: ")
print(f"average val acc: {np.mean(list(domainlabel_target_val_acc_dict.values())):.3f}, average val f1: {np.mean(list(domainlabel_target_val_f1_dict.values())):.3f}")
print(f"average test acc: {np.mean(list(domainlabel_target_test_acc_dict.values())):.3f}, average test f1: {np.mean(list(domainlabel_target_test_f1_dict.values())):.3f}")
target_state = 'IA'
root_dir = '/public12_data/fl/shared/share_mala/llm-dro/income/'
trainx, trainy, valx, valy, testx, testy = load_data('e5', target_state, root_dir, num_train=20000, num_val=32, num_test=5000)

refitx = valx
refity = valy
refity = torch.tensor(refity).long()

domainlabel_model.refit_epochs = 50
domainlabel_model.refit_lr = 0.001
domainlabel_model.refit(refitx, refity)
domainlabel_source_train_acc_dict, domainlabel_source_val_acc_dict, domainlabel_source_test_acc_dict = dict(), dict(), dict()
domainlabel_source_train_f1_dict, domainlabel_source_val_f1_dict, domainlabel_source_test_f1_dict = dict(), dict(), dict()

## report training, val, testing performance for each state
for state in source_state_list:
    root_dir = '/public12_data/fl/shared/share_mala/llm-dro/income/'
    trainx, trainy, valx, valy, testx, testy = load_data('e5', state, root_dir)
    train_acc, train_f1 = domainlabel_model.score(trainx, trainy)
    val_acc, val_f1 = domainlabel_model.score(valx, valy)
    test_acc, test_f1 = domainlabel_model.score(testx, testy)
    
    domainlabel_source_train_acc_dict[state], domainlabel_source_val_acc_dict[state], domainlabel_source_test_acc_dict[state] = train_acc, val_acc, test_acc
    domainlabel_source_train_f1_dict[state], domainlabel_source_val_f1_dict[state], domainlabel_source_test_f1_dict[state] = train_f1, val_f1, test_f1

print("Source State: ")
print(f"average train acc: {np.mean(list(domainlabel_source_train_acc_dict.values())):.3f}, average train f1: {np.mean(list(domainlabel_source_train_f1_dict.values())):.3f}")
print(f"average val acc: {np.mean(list(domainlabel_source_val_acc_dict.values())):.3f}, average val f1: {np.mean(list(domainlabel_source_val_f1_dict.values())):.3f}")
print(f"average test acc: {np.mean(list(domainlabel_source_test_acc_dict.values())):.3f}, average test f1: {np.mean(list(domainlabel_source_test_f1_dict.values())):.3f}")
domainlabel_target_val_acc_dict, domainlabel_target_test_acc_dict =  dict(), dict()
domainlabel_target_val_f1_dict, domainlabel_target_test_f1_dict =  dict(), dict()
for state in ['IA']:
    if state not in source_state_list:
        root_dir = '/public12_data/fl/shared/share_mala/llm-dro/income/'
        trainx, trainy, valx, valy, testx, testy = load_data('e5', state, root_dir)
        val_acc, val_f1 = domainlabel_model.score(valx, valy)
        test_acc, test_f1 = domainlabel_model.score(testx, testy)
        
        domainlabel_target_val_acc_dict[state], domainlabel_target_test_acc_dict[state] = val_acc, test_acc
        domainlabel_target_val_f1_dict[state], domainlabel_target_test_f1_dict[state] = val_f1, test_f1

print("Target State: ")
print(f"average val acc: {np.mean(list(domainlabel_target_val_acc_dict.values())):.3f}, average val f1: {np.mean(list(domainlabel_target_val_f1_dict.values())):.3f}")
print(f"average test acc: {np.mean(list(domainlabel_target_test_acc_dict.values())):.3f}, average test f1: {np.mean(list(domainlabel_target_test_f1_dict.values())):.3f}")