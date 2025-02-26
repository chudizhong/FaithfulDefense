import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import time
from utils import *
from encoder import *
from faithfuldefense import *
from surrogate_model import *
import pickle

palette_exp = {"none": "darkgray", 
               "maximum_coverage_greedy": "orange",
               "maximum_coverage_greedy_all_budget": "red",
               "maximum_coverage_mip": "yellow",
               "maximum_coverage_mip_all_budget": "goldenrod",
               "random": "green",
               "random_all_budget": "lime",
               "base": "blue", 
               "lime": "darkviolet"
            }
labels = {"none": "Baseline no exp",
        "maximum_coverage_greedy_all_budget": "FaithfulDefense Greedy",
        "maximum_coverage_mip": "FaithfulDefense IP",
        "maximum_coverage_mip_all_budget": "FaithfulDefense IP-RA",
        "random_all_budget": "Baseline random exp",
        "base": "Baseline base rule",
        "lime": "Baseline lime"
        }

def plot_supp(dname, max_iter, know_training, query_method, max_len, Nlevels, model_class, seed, metric="test"):
    plt.figure(figsize=(6,6))
    for exp_method in ["none", "base","random_all_budget",  
                    "maximum_coverage_mip", 
                    "maximum_coverage_mip_all_budget",
                    "maximum_coverage_greedy_all_budget"]:
        file = "{}_{}_{}_query_{}_exp_{}_{}_{}_{}_{}".format(dname, max_iter, know_training, query_method, exp_method, max_len, Nlevels, model_class, seed)
        filepath = "results/{}.p".format(file)
        if query_method == "perturb" and exp_method == "none":
            filepath = "results/{}_{}_{}_query_random_exp_{}_{}_{}_{}_{}.p".format(dname, max_iter, know_training, exp_method, max_len, Nlevels, model_class, seed)
            

        surrogate = SurrogateModel(filepath)
        print(len(surrogate.colnames))
        if metric == "train":
            df = pd.read_csv("datasets/{}_train.csv".format(dname))
            X = df.iloc[:,:-1]
            pred_y = find_base_model_prediction(X, surrogate.base_rules)
            exp_supp_count = surrogate.exp_supp_count
        else:
            test = pd.read_csv("datasets/{}_test.csv".format(dname))
            X_test = test.iloc[:,:-1]
            pred_y = find_base_model_prediction(X_test, surrogate.base_rules)
            if exp_method == "none":
                exp_supp_count = np.zeros(surrogate.queries.shape[0])
            else:
                exp_supp_count = surrogate.test_support(X_test)

        label = labels[exp_method]
        plt.plot(np.arange(surrogate.queries.shape[0]), 
                exp_supp_count/pred_y.sum(),
                label=label, color=palette_exp[exp_method], alpha=0.8)
        plt.xlabel("# queries", fontsize=16)
        plt.ylabel("% positive samples \ncovered by explanations", fontsize=16)
        plt.title("{} ({})\n support coverage".format(dname, metric), fontsize=20)
    plt.legend(fontsize=18, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.2))
    plt.tight_layout()
    
    plt.savefig("figures/supp_query_{}_{}_{}_{}_{}.png".format(query_method, max_len, Nlevels, seed, metric), dpi=200, bbox_inches='tight')



def plot_exp_time(dname, max_iter, know_training, max_len, Nlevels, seed):
    plt.figure(figsize=(16.5,3.2))
    exp_times = []
    for query_method in ["random", "iwal", "perturb"]:
        for exp_method in ["base", "random_all_budget", "maximum_coverage_greedy_all_budget", "maximum_coverage_mip", "maximum_coverage_mip_all_budget"]:
            if query_method != "iwal":
                file = "{}_{}_{}_query_{}_exp_{}_{}_{}_none_{}".format(dname, max_iter, know_training, query_method, exp_method, max_len, Nlevels, seed)
            else:
                file = "{}_{}_{}_query_{}_exp_{}_{}_{}_cart_{}".format(dname, max_iter, know_training, query_method, exp_method, max_len, Nlevels, seed)
            filepath = "results/{}.p".format(file)
            if exp_method != "lime":
                surrogate = SurrogateModel(filepath)
            exp_times.append(surrogate.exp_time)
    exp_labels = [labels["base"],  labels["random_all_budget"], 
                  labels["maximum_coverage_greedy_all_budget"], 
                  labels["maximum_coverage_mip"],
                  labels["maximum_coverage_mip_all_budget"]]
    plt.boxplot(exp_times, labels=3*exp_labels)
    plt.xlabel("explanation method", fontsize=15)
    plt.title("{}: explanation time".format(dname), fontsize=18)
    plt.yscale('log')
    plt.ylabel("time", fontsize=15)
    plt.xticks(np.arange(1,3*len(exp_labels)+1), 3*exp_labels, fontsize=13, rotation=45)
    plt.savefig("figures/{}_{}_{}_exp_time_{}.png".format(dname, max_len, Nlevels, seed), dpi=200, bbox_inches='tight')




def plot_compare_onefold(dname, max_iter, know_training, query_method, max_len, Nlevels, seed, model_class, surr_model_class="cart", metric="acc"):
    plt.figure(figsize=(6,6))
    for exp_method in ["none", "base", "random_all_budget", 
                        "maximum_coverage_mip", 
                        "maximum_coverage_mip_all_budget",
                        "maximum_coverage_greedy_all_budget"
                        ]:
        
        label = labels[exp_method]

        if query_method != "iwal":
            file = "{}_{}_{}_query_{}_exp_{}_{}_{}_none_{}".format(dname, max_iter, know_training, query_method, exp_method, max_len, Nlevels, seed)
        else:
            file = "{}_{}_{}_query_{}_exp_{}_{}_{}_{}_{}".format(dname, max_iter, know_training, query_method, exp_method, max_len, Nlevels, model_class, seed)
        
        if query_method == "perturb" and exp_method == "none":
            surrfile = "surrogate_results/surrogate_model_{}_{}_{}_query_{}_exp_{}_{}_{}_none_{}.p".format(dname, max_iter, know_training, "random", "none", max_len, Nlevels, seed)
        else:
            surrfile = "surrogate_results/surrogate_model_{}.p".format(file)
        with open(surrfile, "rb") as f:
            surrout = pickle.load(f)
            
        fidelity_acc = surrout["{}_{}_fidelity_{}".format(surr_model_class, None, metric)]
        plt.plot(np.arange(1, len(fidelity_acc)+1)*surrout["step"], 1-np.array(fidelity_acc),
                    marker=".",
                    label=label, 
                    color=palette_exp[exp_method], alpha=0.95)
               
        plt.xlabel("# queries", fontsize=16)
        plt.ylabel("attacker's error rate ({})".format(surr_model_class), fontsize=16)
        plt.title("{} \ntest performance".format(dname), fontsize=20)
    plt.legend(fontsize=18, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    plt.savefig("figures/surrogate_test_{}_{}_query_{}_{}_{}_compare_{}.png".format(surr_model_class, metric, query_method, max_len, Nlevels, seed), dpi=200, bbox_inches='tight')  





dname = "german_credit"

seed = 0 # 0, 42, 237, 1222, 1757
know_training=False
max_iter = 2000
max_len = 3
Nlevels = 10
model_class = "none" # cart, rf, gbdt, used for iwal, otherwise "none"

query_method = "perturb"
seed = 0


plot_supp(dname, max_iter, know_training, query_method, max_len, Nlevels, model_class, seed, metric="train")
plot_supp(dname, max_iter, know_training, query_method, max_len, Nlevels, model_class, seed, metric="test")

plot_compare_onefold(dname, max_iter, know_training, query_method, max_len, Nlevels, seed, model_class, surr_model_class="cart", metric="acc")
plot_compare_onefold(dname, max_iter, know_training, query_method, max_len, Nlevels, seed, model_class, surr_model_class="rf", metric="acc")
plot_compare_onefold(dname, max_iter, know_training, query_method, max_len, Nlevels, seed, model_class, surr_model_class="gbdt", metric="acc")

plot_exp_time(dname, max_iter, know_training, max_len, Nlevels, seed)
