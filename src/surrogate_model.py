import numpy as np
import pandas as pd
from utils import *
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

class SurrogateModel:
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            out = pickle.load(f)
        self.X = out.X
        self.categorical_indices = out.categorical_indices
        self.continuous_indices = out.continuous_indices
        self.base_rules = out.base_rules
        
        self.enc = out.enc
        self.colnames = out.colnames
        self.exp = out.exp
        self.exp_indices = out.exp_indices # track the iteration the new explanation is appended. 
        self.exp_supp = out.exp_supp # training samples covered by explanations
        self.exp_supp_count = out.exp_supp_count # number of training samples covered by explanations
        self.queries = out.queries
        self.labels = out.labels
        self.exp_time = out.exp_time


    def match_exp(self, query, exp):
        if len(exp) == 0:
            return False
        else:
            query_bv = query_to_bitvec(query, self.enc)
            for e in exp:
                if e!= []:
                    e_bv = cond_indices_to_bitvec(e, self.colnames)
                    if e_bv & query_bv == e_bv:
                        return True
        return False
        

    def find_match(self, X, exp):
        if isinstance(X, np.ndarray):
            match = np.array(list(map(lambda i: self.match_exp(X[i,:], exp), range(X.shape[0]))))
        else:
            match = np.array(list(map(lambda i: self.match_exp(X.iloc[i,:].values, exp), range(X.shape[0]))))
        return match

    def find_binary_samples(self, X, cond_indices, rhs):
        row_idx = X.index[(X.iloc[:, cond_indices] == rhs).all(axis=1)]
        X_select = X.loc[row_idx] 
        return X_select, row_idx 


    def surrogate_model_setup(self, model_class):
        # attacker trains a substitute model
        # model_class: cart, rf, gbdt
        if model_class == "cart":
            f_prime = DecisionTreeClassifier(max_depth=5)    
        elif model_class == "rf":
            f_prime = RandomForestClassifier()
        elif model_class == "gbdt":
            f_prime = GradientBoostingClassifier()     
        return f_prime



    def train_surrogate_model(self, step, model_class, eval_sampling):
        #train a surrogate model and record its performance
        f_prime = self.surrogate_model_setup(model_class)

        queries = self.queries[:step]
        labels = self.labels[:step]
        exp_idx = np.searchsorted(self.exp_indices, step)
        exp = self.exp[:exp_idx]

        if labels.sum() != len(exp):
            print("check length", labels.sum() == len(exp))
            raise Exception("label and explanation don't match")


        count_1 = int(labels.sum())
        count_0 = len(labels) - count_1

        if count_0 == 0 or count_1 == 0:
            return None, exp, count_0, count_1
        
        if count_1 / len(labels) >= 0.4 and count_1 / len(labels) <= 0.6:
            f_prime.fit(queries, labels)
            return f_prime, exp, count_0, count_1
        

        if eval_sampling is None:
            f_prime.fit(queries, labels)
            return f_prime, exp, count_0, count_1

        
       

    def test_support(self, X):
        X_enc = self.enc.transform(X, negation=True)
        exp_supp = np.zeros(X.shape[0])
        exp_supp_count = np.zeros(self.queries.shape[0])

        for i, cond_indices in enumerate(self.exp):
            if len(cond_indices) > 0:
                U, sample_indices = self.find_binary_samples(X_enc, cond_indices, rhs=1) 
                exp_supp[sample_indices] = 1
                exp_supp_count[self.exp_indices[i]] = exp_supp.sum()
            
            # cover points in the middle
            if i > 0:
                exp_supp_count[self.exp_indices[i-1]:self.exp_indices[i]] = exp_supp_count[self.exp_indices[i-1]]
            
            # cover points in the end
            if i == len(self.exp)-1:
                exp_supp_count[self.exp_indices[i]:] = exp_supp_count[self.exp_indices[i]]
        
        return exp_supp_count
    

    def find_exp_cover(self, exps, X):
        # exps must be a list of lists
        X_enc = self.enc.transform(X, negation=True)
        exp_covered = np.zeros(X.shape[0])

        for cond_indices in exps:
            if len(cond_indices) > 0:
                _, sample_indices = self.find_binary_samples(X_enc, cond_indices, rhs=1) 
                exp_covered[sample_indices] = 1
        return exp_covered
            

    
    def main(self, X_test, y_test, step, model_class, eval_sampling):
        pred_y_test = find_base_model_prediction(X_test, self.base_rules)

        
        # calculate the test performance
        steps = np.arange(0, self.queries.shape[0], step)
        steps = steps[1:]
        if steps[-1] != self.queries.shape[0]:
            steps = np.r_[steps, self.queries.shape[0]]
        
        fidelity_accuracy = []
        online_accuracy = []
        fidelity_bacc = []
        fidelity_f1 = []

        time_train = []
        time_eval = []
        count = []
        
        for s in steps:
            s_time = time.time()
            f_prime, exp, count_0, count_1 = self.train_surrogate_model(s, model_class, eval_sampling)
            training_time = time.time() - s_time
            time_train.append(training_time)
            print(s, "model training time:", training_time, flush=True)
            count.append((count_0, count_1))

            
            s_time = time.time()
            if f_prime is None:
                print("---------- f_prime is None")
                acc = accuracy_score(pred_y_test, self.find_exp_cover(exp, X_test))
                fidelity_accuracy.append(acc)
                fidelity_bacc.append(balanced_accuracy_score(pred_y_test, self.find_exp_cover(exp, X_test)))
                fidelity_f1.append(f1_score(pred_y_test, self.find_exp_cover(exp, X_test)))
                

                if s != steps[-1]:
                    online_acc = accuracy_score(self.labels[s:s+step], self.find_exp_cover(exp, pd.DataFrame(self.queries[s:s+step,])))
                    online_accuracy.append(online_acc)
            else:
                exp_covered = self.find_exp_cover(exp, X_test)
                print("exp_covered", exp_covered.sum())
                non_covered_idx = np.where(exp_covered == 0)[0]
                if len(non_covered_idx) > 0:
                    non_covered_pred = f_prime.predict(X_test.iloc[non_covered_idx, ])
                    exp_covered[non_covered_idx] = non_covered_pred

                fidelity_accuracy.append(accuracy_score(pred_y_test, exp_covered))
                fidelity_bacc.append(balanced_accuracy_score(pred_y_test, exp_covered))
                fidelity_f1.append(f1_score(pred_y_test, exp_covered))
                
                
                if s != steps[-1]:
                    exp_covered = self.find_exp_cover(exp, pd.DataFrame(self.queries[s:s+step]))
                    non_covered_idx = np.where(exp_covered == 0)[0]
                    if len(non_covered_idx) > 0:
                        non_covered_pred = f_prime.predict(self.queries[s:s+step][non_covered_idx, ])
                        exp_covered[non_covered_idx] = non_covered_pred
                    online_acc = accuracy_score(self.labels[s:s+step], exp_covered)
                    online_accuracy.append(online_acc)
            eval_time = time.time() - s_time
            time_eval.append(eval_time)
            print(s, "model evaluation time:", eval_time, flush=True)
        return fidelity_accuracy, fidelity_bacc, fidelity_f1, online_accuracy, count, time_train, time_eval
    
    def exp_pred(self,  X_test, step):
        pred_y_test = find_base_model_prediction(X_test, self.base_rules)
        X_test_enc = self.enc.transform(X_test, negation=True)

        test_accuracy = []
        online_accuracy = []
        test_bacc = []
        test_f1 = []
        test_fp = []
        test_fn = []
        
        s_time = time.time()

        yhat = np.zeros(X_test.shape[0])
        for j in range(len(self.exp)):
            exp = self.exp[j]
            if len(exp) > 0:
                _, sample_indices = self.find_binary_samples(X_test_enc, exp, rhs=1) 
                yhat[sample_indices] = 1
            tn, fp, fn, tp = confusion_matrix(pred_y_test, yhat).ravel()
            test_fp.append(fp)
            test_fn.append(fn)
            test_accuracy.append(accuracy_score(pred_y_test,yhat))
            test_bacc.append(balanced_accuracy_score(pred_y_test, yhat))
            test_f1.append(f1_score(pred_y_test, yhat))
        time_eval = time.time() - s_time
        return test_accuracy, test_bacc, test_f1, test_fp, test_fn, time_eval

     