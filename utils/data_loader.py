import numpy as np
import pandas as pd
from reduce_memory import reduce_mem_usage
from feature_process import drop_anormal

def load_train_data(name):
    train = pd.read_csv(f"data/{name}")
    
    train = drop_anormal(train)

    train = reduce_mem_usage(train)

    return train

def load_test_data(name):

    test = pd.read_csv(f"data/{name}")
    
    test = reduce_mem_usage(test)
    return test

def load_submission_data(name):

    submission = pd.read_csv(f"data/{name}")
 
    return submission