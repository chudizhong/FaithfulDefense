import numpy as np
import pandas as pd
from bitarray import bitarray, util
import time
import pickle
import warnings
from encoder import *

def transform_base_rules(base_rules, X):
    columns = list(X.columns)
    valid_base_rules = []
    pair = {}
    support = []
    for rule in base_rules:
        rule = rule.split(" AND ")
        condition = []
        X_init = X.copy()
        for att in rule:
            if " == " in att:
                f, tr = att.split(" == ")
                f_idx = columns.index(f)
                condition.append("{} == {}".format(f_idx, float(tr)))
                X_init = X_init[X_init[f] == float(tr)]
            elif " != " in att:
                f, tr = att.split(" != ")
                f_idx = columns.index(f)
                condition.append("{} != {}".format(f_idx, float(tr)))
                X_init = X_init[X_init[f] != float(tr)]
            elif " <= " in att:
                f, tr = att.split(" <= ")
                f_idx = columns.index(f)
                condition.append("{} <= {}".format(f_idx, float(tr)))
                X_init = X_init[X_init[f] <= float(tr)]
                if f_idx not in pair:
                    pair[f_idx] = [float(tr)]
                else:
                    pair[f_idx].append(float(tr))   
            elif " > " in att:
                f, tr = att.split(" > ")
                f_idx = columns.index(f)
                condition.append("{} > {}".format(f_idx, float(tr)))
                X_init = X_init[X_init[f] > float(tr)]
                if f_idx not in pair:
                    pair[f_idx] = [float(tr)]
                else:
                    pair[f_idx].append(float(tr))   
            
                    
        support.append(X_init.shape[0])
        valid_base_rules.append(condition)
    for k, v in pair.items():
        pair[k] = sorted(np.unique(v))
    pair = dict(sorted(pair.items()))
    return valid_base_rules, pair, support


def get_prediction(query_bv, base_rules_bv):
    for r in base_rules_bv:
        if r & query_bv == r:
            return 1
    return 0



def query_to_bitvec(query, enc):   
    # query is a numpy array
    bv = enc.transform(pd.DataFrame(query.reshape(1,-1)), negation=True)
    bv = bitarray("".join(map(str, bv.values[0])))
    return bv

  

def cond_indices_to_bitvec(cond_indices, colnames):
    exp_bv = np.zeros(len(colnames), dtype=int)
    exp_bv[cond_indices] = 1
    return bitarray("".join(map(str, exp_bv))) 
    

def rule_to_bitvec(rule, colnames):
    """
    rule: [['0 <= 73.0', '14 <= 0.0'], ['17 > 43.0', '4 <= 11.0']]
    """
    rule_bv = np.zeros(len(colnames), dtype=int)
    for r in rule:
        rule_bv[colnames.index(r)]=1
        
    return bitarray("".join(map(str, rule_bv))) 


def find_base_model_prediction(X, base_rules):   
    pred = np.zeros(X.shape[0])     
    for rule in base_rules:
        X_copy = X.copy()
        for r in rule:
            if " == " in r:
                f, tr = r.split(" == ")
                f_idx = int(f)
                X_copy = X_copy[X_copy.iloc[:,f_idx]==float(tr)]
            elif " != " in r:
                f, tr = r.split(" != ")
                f_idx = int(f)
                X_copy = X_copy[X_copy.iloc[:, f_idx] != float(tr)]
            elif " <= " in r:
                f, tr = r.split(" <= ")
                f_idx = int(f)
                X_copy = X_copy[X_copy.iloc[:, f_idx] <= float(tr)]
            elif " > " in r:
                f, tr = r.split(" > ")
                f_idx = int(f)
                X_copy = X_copy[X_copy.iloc[:, f_idx] > float(tr)]
              
        row_idx = X_copy.index # row index 
        pred[row_idx] = 1
    return pred

def transform_exp(exp, colnames, categorical_indices, continuous_indices):
    used_colnames = np.array(colnames)[np.array(exp)]
    transform_colnames = np.array([i.split(" ") for i in used_colnames])
    used_features = np.array(sorted(np.unique(transform_colnames[:,0].astype(int))))
    
    # initialization
    valid_e = {}
    for f in used_features:
        if f in categorical_indices:
            valid_e[f] = -1
        else:
            valid_e[f] = [-float("Inf"), float("Inf")]
    
    for f in used_features:
        rows = transform_colnames[np.where(transform_colnames[:,0] == str(f))[0]]
        for i in range(rows.shape[0]):
            if f in categorical_indices:
                valid_e[f] = int(float(rows[i,2]))
            else:
                if rows[i,1] == "<=":
                    if valid_e[f][1] == float("Inf"):
                        valid_e[f][1] = min(float(rows[i,2]), valid_e[f][1])
                    else:
                        valid_e[f][1] = max(float(rows[i,2]), valid_e[f][1])
                elif rows[i,1] == ">":
                    if valid_e[f][0] == -float("Inf"):
                        valid_e[f][0] = max(float(rows[i,2]), valid_e[f][0])
                    else:
                        valid_e[f][0] = min(float(rows[i,2]), valid_e[f][0])
                    
    return valid_e, used_colnames, used_features
    
