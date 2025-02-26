import sys
sys.path.append('/usr/pkg/cplex-studio-221/cplex/python/3.10/x86-64_linux/')
import cplex
from cplex.exceptions import CplexError
import numpy as np
import pandas as pd
from bitarray import bitarray, util
import time
import warnings
from utils import *
from encoder import *
from sklearn.tree import DecisionTreeClassifier, plot_tree
import copy
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

class AttackDefend:
    def __init__(self, X, ft_split, base_rules, max_len, know_training=False, Nlevels=10, seed=0):
        self.X = X
        self.categorical_indices = [i for i in range(ft_split)]
        self.continuous_indices = [i for i in range(ft_split, X.shape[1])]
        
        base_rules, ft_pair, base_rules_support_size = transform_base_rules(base_rules, self.X)
        # order base rules, smaller support first
        # skip base rule that covers no samples
        self.base_rules = [r for _, r in sorted(zip(base_rules_support_size, base_rules)) if _ > 0] 
        
        
        self.enc = encoder(Nlevels = Nlevels)
        self.enc.fit(self.X, self.categorical_indices, self.continuous_indices, ft_pair)
        self.X_enc = self.enc.transform(X, negation=True)
        self.p = int(self.X_enc.shape[1])
        self.colnames = list(self.X_enc.columns)
        
        self.base_rules_bv = []
        for rule in self.base_rules:
            col_init = rule_to_bitvec(rule, self.colnames)
            self.base_rules_bv.append(col_init)
        
        self.max_len = max_len
        self.know_training = know_training
        self.seed = seed
        self.exp = []
        self.exp_indices = [] # track the iteration the new explanation is appended. 
        self.exp_supp = np.zeros(self.X.shape[0]) # training samples covered by explanations
        self.exp_supp_count = np.array([]) # number of training samples covered by explanations
        self.queries = np.array([]).reshape(0, X.shape[1])
        self.labels = np.array([])
        self.probs = np.array([])
        self.queue = []
        self.feature_count = dict.fromkeys(range(self.X.shape[1]), 0)

        self.exp_time = []
    
    
    def find_binary_samples(self, X, cond_indices, rhs):
        row_idx = X.index[(X.iloc[:, cond_indices] == rhs).all(axis=1)]
        X_select = X.loc[row_idx] 
        return X_select, row_idx 
    
    
    def get_tight_exp_bound(self, x, cond):
        if " <= " in cond:
            f, tr = cond.split(" <= ")
            f, tr = int(f), float(tr)
            x_tr = float(x[f])
            if x_tr in self.enc.continuous_thresholds_in_dic[f]:
                c_idx = self.colnames.index("{} <= {}".format(f, x_tr))
            else:
                tr_idx = np.searchsorted(self.enc.continuous_thresholds_in_dic[f], x_tr)
                c_idx = self.colnames.index("{} <= {}".format(f, self.enc.continuous_thresholds_in_dic[f][tr_idx]))
            
        elif " > " in cond:
            f, tr = cond.split(" > ")
            f, tr = int(f), float(tr)
            x_tr = float(x[f])
            if x_tr in self.enc.continuous_thresholds_in_dic[f]:
                c_idx = self.colnames.index("{} > {}".format(f, x_tr))
            else:
                tr_idx = np.searchsorted(self.enc.continuous_thresholds_in_dic[f], x_tr)-1
                c_idx = self.colnames.index("{} > {}".format(f, self.enc.continuous_thresholds_in_dic[f][tr_idx]))
                    
        return c_idx
    
    def greedy_method(self, U_neg, sample_indices, cond_indices, extra_cond_indices):
        for i in range(self.max_len):
            if U_neg.empty:
                break
            weight_vec = self.exp_supp[sample_indices]
            weight_column = U_neg.values[:, extra_cond_indices]*(1-weight_vec)[:, np.newaxis]
            count = weight_column.sum(axis=0)
            cond_indices.append(extra_cond_indices[np.random.choice(np.argwhere(count == np.amax(count)).flatten())])
            U_neg, sample_indices = self.find_binary_samples(U_neg, cond_indices, rhs=0)
        return cond_indices
    
    def greedy_method_all_budget(self, U_neg, sample_indices, cond_indices, extra_cond_indices):
        init_len = len(cond_indices)
        cond_indices = self.greedy_method(U_neg, sample_indices, cond_indices, extra_cond_indices)
        
        # randomly append extra conditions to use all max_len budget
        if len(cond_indices) < init_len + self.max_len:
            extra_cond_indices = np.setdiff1d(extra_cond_indices, np.array(cond_indices))
            cond_indices += np.random.choice(extra_cond_indices, init_len+self.max_len - len(cond_indices), replace=False).tolist()     
        return cond_indices
    
    def mip_method(self, U_neg, sample_indices, cond_indices, extra_cond_indices):
        weight_vec = self.exp_supp[sample_indices]
        len_u = len(weight_vec)
        len_v = len(extra_cond_indices)
        coef_obj = (1-weight_vec).tolist() + [0.0]*len_v
        var_ub = [1]*(len_u + len_v)
        var_lb = [0]*(len_u + len_v)
        var_type = 'I'*(len_u + len_v)
        var_names = ["u{}".format(i) for i in range(len_u)] + ["v{}".format(j) for j in range(len_v)] 

        rhs = [self.max_len] + [0.0]*len_u  
        sense = "L" + "G"*len_u 
        cst = [[np.arange(len_u, len_u+len_v).tolist(), [1]*len_v]] # no more than len_v conditions are selected
        for i in range(len_u):
            c_idx = np.where(U_neg.iloc[i,extra_cond_indices] == 1)[0]
            cst_coef = np.zeros(len_v)
            cst_coef[c_idx] = 1
            cst += [[np.arange(len_u, len_u+len_v).tolist() + [i], cst_coef.tolist() + [-1]]]
        cst_names = ["max_select"] + ["cover_"+str(i) for i in range(len_u)]

        model = cplex.Cplex()
        model.parameters.timelimit.set(180)
        model.parameters.randomseed.set(self.seed)
        model.parameters.emphasis.memory.set(True)
        model.parameters.mip.tolerances.mipgap.set(1e-5)
        
        model.set_log_stream(None)
        model.set_error_stream(None)
        model.set_warning_stream(None)
        model.set_results_stream(None)
            
        start_time = time.time()
            
        model.objective.set_sense(model.objective.sense.maximize)
        model.variables.add(obj=coef_obj, lb=var_lb, ub=var_ub, 
                            types=var_type, names=var_names)
        model.linear_constraints.add(lin_expr=cst, senses=sense, 
                                        rhs=rhs, names=cst_names)
        
        f_time = time.time()-start_time
        print('seconds formulating problem:', f_time, flush=True)
            
        model.solve()
        
        s_time = time.time()-f_time - start_time
        print('solving time:', s_time, flush=True)

        print("Solution status = ", model.solution.get_status(), ":", end=' ')
        print(model.solution.status[model.solution.get_status()])
        print("Solution value  = ", model.solution.get_objective_value())
       
        var = model.solution.get_values()
        us = var[:len_u]
        vs = var[len_u:(len_u+len_v)] # length of len_v
        vs_round = [round(i) for i in vs]
        cond_add = extra_cond_indices[np.where(np.array(vs_round)==1)[0]].tolist()
        print("sum vs", sum(vs),  len(cond_add), flush=True)
        cond_indices += cond_add 
        return cond_indices
    
    def mip_method_all_budget(self, U_neg, sample_indices, cond_indices, extra_cond_indices):
        init_len = len(cond_indices)
        cond_indices = self.mip_method(U_neg, sample_indices, cond_indices, extra_cond_indices)

        # randomly append extra conditions to use all max_len budget
        if len(cond_indices) < init_len + self.max_len:
            extra_cond_indices = np.setdiff1d(extra_cond_indices, np.array(cond_indices))
            cond_indices += np.random.choice(extra_cond_indices, init_len+self.max_len - len(cond_indices), replace=False).tolist()     
        return cond_indices
        
    
    def generate_maximum_coverage_exp(self, x, mc_method):
        x_bv = query_to_bitvec(x, self.enc)
        
        # find all matched base rules
        matched_base_rules = []
        for rule_idx, rule in enumerate(self.base_rules_bv):
            if rule & x_bv == rule:
                matched_base_rules.append(rule_idx)
        
                
        # find an explanation with minimum support 
        best_supp = np.ones(self.X.shape[0])
        best_cond = []
        for rule_idx in matched_base_rules:
            rule = self.base_rules[rule_idx]
            
            # get tight bound for each condition in the rule
            cond_indices = []
            for cond in rule:
                if (" == " in cond) or (" != " in cond):
                    cond_indices.append(self.colnames.index(cond))
                else:
                    cond_indices.append(self.get_tight_exp_bound(x, cond))
            
            # add more conditions
            extra_cond_indices = np.where(self.enc.transform(pd.DataFrame(x.reshape(1,-1))).values.flatten() == 1)[0]
            U, sample_indices = self.find_binary_samples(self.X_enc, cond_indices, rhs=1) 
            U_neg = abs(U-1)  
            
            # greedy method or mip method
            if mc_method == "greedy":
                cond_indices = self.greedy_method(U_neg, sample_indices, cond_indices, extra_cond_indices)
            elif mc_method == "greedy_all_budget":
                cond_indices = self.greedy_method_all_budget(U_neg, sample_indices, cond_indices, extra_cond_indices)
            elif mc_method == "mip":
                cond_indices = self.mip_method(U_neg, sample_indices, cond_indices, extra_cond_indices)
            elif mc_method == "mip_all_budget":
                cond_indices = self.mip_method_all_budget(U_neg, sample_indices, cond_indices, extra_cond_indices)
            else:
                raise ValueError("only support 'greedy' method and 'mip' method")
            
            
            U, sample_indices = self.find_binary_samples(self.X_enc, cond_indices, rhs=1)
            if len(sample_indices) < len(best_supp):
                best_supp = sample_indices
                best_cond = cond_indices
            
        self.exp.append(sorted(best_cond)) 
        self.exp_supp[best_supp] = 1
    
    
    def generate_base_exp(self, x):
        x_bv = query_to_bitvec(x, self.enc)
        
        # find the first matched base rule
        for rule_idx, rule in enumerate(self.base_rules_bv):
            if rule & x_bv == rule:
                break
        cond_indices = [self.colnames.index(cond) for cond in self.base_rules[rule_idx]]
        U, sample_indices = self.find_binary_samples(self.X_enc, cond_indices, rhs=1)
            
        self.exp.append(sorted(cond_indices))
        self.exp_supp[sample_indices] = 1
    
    
    def generate_random_exp(self, x):
        x_bv = query_to_bitvec(x, self.enc)
        
        # find all matched base rules
        matched_base_rules = []
        for rule_idx, rule in enumerate(self.base_rules_bv):
            if rule & x_bv == rule:
                matched_base_rules.append(rule_idx)
        
        # randomly pick a matched base rule from all matched base rules
        rule_idx = np.random.choice(matched_base_rules)
        rule = self.base_rules[rule_idx]
        cond_indices = [self.colnames.index(cond) for cond in rule]
        
        # randomly append at most k conditions
        x_cond_indices = np.where(self.enc.transform(pd.DataFrame(x.reshape(1,-1))).values.flatten() == 1)[0]
        extra_cond_indices = np.array([i for i in x_cond_indices if i not in cond_indices])
        U, sample_indices = self.find_binary_samples(self.X_enc, cond_indices, rhs=1) 
        for i in range(self.max_len):
            if U.shape[0]==1:
                break
            cond_idx = np.random.choice(extra_cond_indices)
            cond_indices.append(cond_idx)
            U, sample_indices = self.find_binary_samples(U, cond_indices, rhs=1)
            extra_cond_indices = np.delete(extra_cond_indices, np.where(extra_cond_indices == cond_idx))

        U, sample_indices = self.find_binary_samples(self.X_enc, cond_indices, rhs=1)
        self.exp.append(sorted(cond_indices))
        self.exp_supp[sample_indices] = 1
    

    def generate_random_exp_all_budget(self, x):
        x_bv = query_to_bitvec(x, self.enc)
        
        # find all matched base rules
        matched_base_rules = []
        for rule_idx, rule in enumerate(self.base_rules_bv):
            if rule & x_bv == rule:
                matched_base_rules.append(rule_idx)
        
        # randomly pick a matched base rule from all matched base rules
        rule_idx = np.random.choice(matched_base_rules)
        rule = self.base_rules[rule_idx]
        cond_indices = [self.colnames.index(cond) for cond in rule]
        
        # randomly append k conditions
        x_cond_indices = np.where(self.enc.transform(pd.DataFrame(x.reshape(1,-1))).values.flatten() == 1)[0]
        extra_cond_indices = np.array([i for i in x_cond_indices if i not in cond_indices])
        cond_indices += np.random.choice(extra_cond_indices, self.max_len, replace=False).tolist()     

        U, sample_indices = self.find_binary_samples(self.X_enc, cond_indices, rhs=1)
        self.exp.append(sorted(cond_indices))
        self.exp_supp[sample_indices] = 1
    
    def generate_single_exp(self, x, exp_method):
        start_time = time.time()
        if exp_method == "base":
            self.generate_base_exp(x)
        elif exp_method == "random":
            self.generate_random_exp(x)
        elif exp_method == "random_all_budget":
            self.generate_random_exp_all_budget(x)
        elif exp_method == "maximum_coverage_greedy":
            self.generate_maximum_coverage_exp(x, "greedy")
        elif exp_method == "maximum_coverage_greedy_all_budget":
            self.generate_maximum_coverage_exp(x, "greedy_all_budget")
        elif exp_method == "maximum_coverage_mip":
            self.generate_maximum_coverage_exp(x, "mip")
        elif exp_method == "maximum_coverage_mip_all_budget":
            self.generate_maximum_coverage_exp(x, "mip_all_budget")
        elif exp_method == "none":
            #no explanation given
            self.exp.append([]) # append an empty list
        else:
            raise ValueError("only support base, random, maximum_coverage explanation methods")

        duration = time.time() - start_time
        self.exp_time.append(duration) 
      
    
    def surrogate_model(self, queries, labels, probs, model_class):
        # attacker trains a substitute model
        # model_class: cart, rf, gbdt
        if model_class == "cart":
            f_prime = DecisionTreeClassifier(max_depth=5)    
        elif model_class == "rf":
            f_prime = RandomForestClassifier()
        elif model_class == "gbdt":
            f_prime = GradientBoostingClassifier()
        
        if len(probs) == 0:
            f_prime.fit(queries, labels)
        else:
            f_prime.fit(queries, labels, sample_weight=1/probs)
        return f_prime
    
    
    def match_exp(self, exp, query):
        if len(exp) == 0:
            return False
        else:
            query_bv = query_to_bitvec(query, self.enc)
            for e in exp:
                # add if condition in case of empty explanation
                if e!= []:
                    e_bv = cond_indices_to_bitvec(e, self.colnames)
                    if e_bv & query_bv == e_bv:
                        return True
        return False
        

             
    def random_query(self, n=1):
        # generate a pool of query in np array
        if self.know_training: 
            idx = np.random.choice(self.X.shape[0], n)
            queries = self.X.iloc[idx, :].values
        else:
            queries = np.array([]).reshape(n, 0)
            for c in self.X.columns:
                queries = np.c_[queries, np.random.choice(self.X[c], n)]
        return queries
    
    
    def find_unique_query_outside_exp(self):
        select_query = False
        while select_query is False:
            q = self.random_query(1)[0]
            if (self.match_exp(self.exp, q) is False) and (q.tolist() not in self.queries.tolist()):
                select_query = True
        return q
    
    
    def iwal_t(self, model_class, t, c0=8, c1=1, c2=1):
        # model_class: cart, rf, gbdt
        # get an unlabeled query outside the union of explanations
        q = self.find_unique_query_outside_exp()
        
        f_prev = self.surrogate_model(self.queries, self.labels, self.probs, model_class) 
        yhat = f_prev.predict(q.reshape(1,-1))[0] # yhat = {0,1}
        
        yflip = abs(yhat-1)
        queries = np.r_[self.queries, q.reshape(1,-1)]
        labels = np.r_[self.labels, yflip]
        probs = np.r_[self.probs, min(self.probs)/10]
        
        flip_label = False
        while flip_label == False:
            for i in range(10):
                f_prev_prime = self.surrogate_model(queries, labels, probs, model_class)
                if f_prev_prime.predict(q.reshape(1,-1)) == yflip:
                    flip_label = True
                    break
            probs[-1] /= 10
        G =  f_prev.score(self.queries, self.labels, self.probs) - f_prev_prime.score(self.queries, self.labels, self.probs)
        term2 = c0 * np.log(t) / (t-1)
        term1 = np.sqrt(term2)
        reject_thresh =  term1 + term2
        if t % 100 == 0:
            print("G={}, reject_thresh={}".format(G, reject_thresh), flush=True)
        if G <= reject_thresh:
            prob_t = 1
        else:
            A = G - (1-c1)*term1 - (1-c2)*term2
            B = -c1*term1
            C = -c2*term2
            s_sqrt = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
            prob_t = s_sqrt**2
            if prob_t >= 1 or prob_t <= 0:
                warnings.warn("probability is not in (0,1)")
        
        ask_query = np.random.binomial(1, prob_t)
        if ask_query == 1:
            return t, prob_t, q
        else:
            return t, None, None
    
    def perturb_feature(self, q, exp, k=3):
        queue_new = []

        valid_exp, used_colnames, used_features = transform_exp(exp, self.colnames, self.categorical_indices, self.continuous_indices)
        top_k_indices = np.argsort([self.feature_count.get(i) for i in used_features])[::-1][:k][::-1]
        select_features = used_features[top_k_indices]
        
        # generate queries based on select_features
        for f in select_features:
            if f in self.categorical_indices:
                categories = np.unique(self.X.iloc[:,f])
                if valid_exp[f]-1 in categories:
                    q_new = q.copy()
                    q_new[f] = valid_exp[f]-1
                    queue_new.append((q_new, f))
                
                if valid_exp[f]+1 in categories:
                    q_new = q.copy()
                    q_new[f] = valid_exp[f] + 1
                    queue_new.append((q_new, f))
                
            else:
                if valid_exp[f][1] != float("Inf") and valid_exp[f][0] != -float("Inf"):
                    q_1, q_2 = q.copy(), q.copy()
                    q_1[f] = valid_exp[f][1] + 1
                    q_2[f] = valid_exp[f][0] - 1
                    queue_new.append((q_1, f))
                    queue_new.append((q_2, f))

                elif valid_exp[f][1] != float("Inf") and valid_exp[f][0] == -float("Inf"):
                    q_new = q.copy()
                    q_new[f] = valid_exp[f][1] + 1
                    queue_new.append((q_new, f))
                elif valid_exp[f][1] == float("Inf") and valid_exp[f][0] != -float("Inf"):
                    q_new = q.copy()
                    q_new[f] = valid_exp[f][0] - 1
                    queue_new.append((q_new, f))
        
        self.queue = [(x[0], x[1], self.feature_count[x[1]]) for x in self.queue if (self.match_exp([exp], x[0]) is False) and (x[0].tolist() not in self.queries.tolist())]
        queue_new = [(x[0], x[1], self.feature_count[x[1]]) for x in queue_new if (self.match_exp(self.exp, x[0]) is False) and (x[0].tolist() not in self.queries.tolist())]
        self.queue += queue_new
        self.queue = sorted(self.queue, key = lambda x: x[2], reverse=True)
        # update self.feature_count based on the exp
        for f in used_features:
            self.feature_count[f] += 1

    def init_rounds(self, exp_method, init_iter):
        for i in range(init_iter):
            q = self.find_unique_query_outside_exp()
            yhat = get_prediction(query_to_bitvec(q, self.enc), self.base_rules_bv)
            self.queries = np.r_[self.queries, q.reshape(1,-1)]
            self.labels = np.r_[self.labels, yhat]
                
            if yhat == 1:
                self.generate_single_exp(q, exp_method)
                self.exp_indices.append(i)
            self.exp_supp_count = np.r_[self.exp_supp_count, self.exp_supp.sum()]



    def main(self, query_method, exp_method, max_iter, init_iter=0, model_class="cart", n=None):
        # query method = {random, iwal, perturb}
        # explanation method = {maximum_coverage_greedy, maximum_coverage_mip, random, base, none}
        # model class = {cart, rf, gbdt}

        # if query_method is based on active learning and init_iter > 0
        if query_method == "iwal" and init_iter != 0:
            self.init_rounds(exp_method, init_iter)
            if query_method == "iwal":
                t = init_iter
                self.probs = np.r_[self.probs, np.ones(init_iter)]
        

        for i in range(init_iter, max_iter):
            if i % 100 == 0:
                print(i, flush=True)

            if query_method == "random":
                q = self.find_unique_query_outside_exp()
            elif query_method == "iwal":
                q = None
                while q is None:
                    t, prob_t, q = self.iwal_t(model_class, t+1)
                self.probs = np.r_[self.probs, prob_t]
            elif query_method == "perturb":
                if len(self.queue) > 0:
                    q = self.queue.pop(0)[0]
                else: 
                    q = self.find_unique_query_outside_exp()
                
            
            yhat = get_prediction(query_to_bitvec(q, self.enc), self.base_rules_bv)
            self.queries = np.r_[self.queries, q.reshape(1,-1)]
            self.labels = np.r_[self.labels, yhat]
            
            if yhat == 0:
                self.exp_supp_count = np.r_[self.exp_supp_count, self.exp_supp.sum()]
                continue
            
            if self.match_exp(self.exp, q) is True:
                self.exp_supp_count = np.r_[self.exp_supp_count, self.exp_supp.sum()]
                continue
            
            # generate an explanation
            self.generate_single_exp(q, exp_method)
            self.exp_indices.append(i)
            if query_method == "perturb":
                self.perturb_feature(q, self.exp[-1])
            self.exp_supp_count = np.r_[self.exp_supp_count, self.exp_supp.sum()]
            

       