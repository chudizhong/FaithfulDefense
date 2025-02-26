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
from surrogate_model import *


dname = "german_credit"
test = pd.read_csv("datasets/{}_test.csv".format(dname))
X_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1].values


seed = 0 
know_training=False
max_iter = 2000
max_len = 3
Nlevels = 10
model_class = "none" # cart used for iwal, otherwise "none"

query_method = "random" # random, perturb, iwal
for exp_method in ["none", "base", "random_all_budget", "maximum_coverage_greedy_all_budget", "maximum_coverage_mip", "maximum_coverage_mip_all_budget"]:
    if query_method == "perturb" and exp_method == "none":
        continue
    file = "{}_{}_{}_query_{}_exp_{}_{}_{}_{}_{}".format(dname, max_iter, know_training, query_method, exp_method, max_len, Nlevels, model_class, seed)
    filepath = "results/{}.p".format(file)
    print(filepath, flush=True)

    np.random.seed(seed)
    random.seed(seed)


    step = 50

    results = {
                "date": time.strftime("%d/%m/%y", time.localtime()),
                "data_file": dname,
                "step": step
            }  

    outfile = "surrogate_results/surrogate_model_{}.p".format(file)

    with open(outfile, "wb") as out:
        pickle.dump(results, out, protocol=pickle.DEFAULT_PROTOCOL)

    for surr_model_class in ["cart", "rf", "gbdt"]:
        eval_sampling = None
        print(surr_model_class, eval_sampling, flush=True)
        start_time = time.time()
        surrogate = SurrogateModel(filepath)
        fidelity_accuracy, fidelity_bacc, fidelity_f1, online_accuracy, count, time_train, time_eval = surrogate.main(X_test, y_test, step, surr_model_class, eval_sampling)
        duration = time.time() - start_time
        print(surr_model_class, eval_sampling, "surrogate model time:", duration)
        
        with open(outfile, "rb") as f:
            results = pickle.load(f)
        
        results["{}_{}_fidelity_acc".format(surr_model_class, eval_sampling)] = fidelity_accuracy
        results["{}_{}_fidelity_bacc".format(surr_model_class, eval_sampling)] = fidelity_bacc
        results["{}_{}_fidelity_f1".format(surr_model_class, eval_sampling)] = fidelity_f1
        
        results["{}_{}_online_acc".format(surr_model_class, eval_sampling)] = online_accuracy
        results["{}_{}_count".format(surr_model_class, eval_sampling)] = count
        results["{}_{}_time_train".format(surr_model_class, eval_sampling)] = time_train
        results["{}_{}_time_eval".format(surr_model_class, eval_sampling)] = time_eval

        with open(outfile, "wb") as out:
            pickle.dump(results, out, protocol=pickle.DEFAULT_PROTOCOL)

    if exp_method != "none":
        with open(outfile, "rb") as f:
            results = pickle.load(f)
        surrogate = SurrogateModel(filepath)
        test_accuracy, test_bacc, test_f1, test_fp, test_fn,  time_eval = surrogate.exp_pred(X_test, step)
        results["exp_only_test_acc"] = test_accuracy
        results["exp_only_test_bacc"] = test_bacc
        results["exp_only_test_f1"] = test_f1
        results["exp_only_test_fp"] = test_fp
        results["exp_only_test_fn"] = test_fn
        results["exp_only_time_eval"] = time_eval

        with open(outfile, "wb") as out:
            pickle.dump(results, out, protocol=pickle.DEFAULT_PROTOCOL)

