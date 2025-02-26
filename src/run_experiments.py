import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from utils import *
from encoder import *
from faithfuldefense import *
import pickle
import json
import random

dname = "german_credit"
ft_split = 13

df = pd.read_csv("datasets/{}_train.csv".format(dname))
X = df.iloc[:,:-1]

test = pd.read_csv("datasets/{}_test.csv".format(dname))
X_test = test.iloc[:,:-1]

with open("datasets/base_rules.json") as f:
    base_rules = json.load(f)
base_rules = base_rules[dname]

seed = 0 
know_training=False
max_iter = 2000
init_iter = 0 # default 0, iwal 50

max_len = 3
Nlevels = 10
model_class = "none" # cart used for iwal

query_method = "random" # random, perturb, iwal
for exp_method in ["none", "base", "random_all_budget", 
                   "maximum_coverage_greedy_all_budget",
                   "maximum_coverage_mip",
                   "maximum_coverage_mip_all_budget"]:
    if query_method == "perturb" and exp_method == "none":
        continue
    outfile = "results/{}_{}_{}_query_{}_exp_{}_{}_{}_{}_{}.p".format(dname, max_iter, know_training, query_method, exp_method, max_len, Nlevels, model_class, seed)
    print(outfile)

    np.random.seed(seed)
    random.seed(seed)

    start_time = time.time()
    model = AttackDefend(X, ft_split, base_rules, max_len, know_training, Nlevels, seed)
    model.main(query_method, exp_method, max_iter, init_iter, model_class)
    duration = time.time() - start_time
    print("total time:", duration)


    with open(outfile, "wb") as out:
        pickle.dump(model, out, protocol=pickle.DEFAULT_PROTOCOL)