import numpy as np
import pandas as pd

class encoder():
    def __init__(self, Nlevels: int=10):
        self.Nlevels = Nlevels
        
        
    def fit(self, X, categorical_indices, continuous_indices, ft_pair):
        categorical_thresholds = []
        continuous_thresholds = []
        continuous_thresholds_in_dic = {}
        for j in range(X.shape[1]):
            if j in categorical_indices:
                categorical_thresholds += [(j, i) for i in np.unique(X.iloc[:,j])]
            else:
                if j in ft_pair:
                    thresh = ft_pair[j]
                    n = self.Nlevels - len(thresh)
                    count = []
                    for k in range(len(thresh)):
                        if k == 0:
                            n_sample = sum(X.iloc[:,j] <= thresh[k])
                        else:
                            n_sample = sum((X.iloc[:,j] > thresh[k-1]) & (X.iloc[:,j] <= thresh[k]))
                        count.append(n_sample)
                        
                    count.append(sum(X.iloc[:,j] > thresh[k]))
                    count = np.array(count)
                    ns = np.round(n*count/X.shape[0])
                    prop = np.r_[0, np.cumsum(count/X.shape[0])]
                    qt_thresh = np.array([])
                    for i in range(1, len(prop)):
                        qt_thresh = np.r_[qt_thresh, np.round(X.iloc[:,j].quantile(np.linspace(prop[i-1], prop[i], int(ns[i-1])+1, endpoint=False)[1:]).values)]
                        if i != len(prop)-1:
                            qt_thresh = np.r_[qt_thresh, thresh[i-1]]
                else:
                    qt_thresh = np.round(X.iloc[:,j].quantile(np.linspace(0.1, 1, self.Nlevels, endpoint=False)).values)
                qt_thresh = np.unique(qt_thresh)
            
                # print(qt_thresh)
                continuous_thresholds += [(j, i) for i in qt_thresh]
                continuous_thresholds_in_dic[j] = qt_thresh
        
        self.categorical_thresholds = categorical_thresholds
        
        self.continuous_thresholds = continuous_thresholds
        self.continuous_thresholds_in_dic = continuous_thresholds_in_dic
        return self
    

    def transform(self, X, negation=True):
        feature_names_out = []
        X_init = np.atleast_2d(X.values)
        X_new = np.array([]).reshape(X_init.shape[0], -1)
        
        if len(self.categorical_thresholds) > 0:
            feature_names_out += [f'{j} == {float(thresh)}' for j, thresh in self.categorical_thresholds]
            X_categorical_new = np.concatenate([np.atleast_2d(X_init[:, j] == thresh).T for j, thresh in self.categorical_thresholds], axis=1).astype(int)
            X_new = np.c_[X_new, X_categorical_new]
        
        if len(self.continuous_thresholds) > 0:
            feature_names_out += [f'{j} <= {thresh}' for j, thresh in self.continuous_thresholds] 
            X_continuous_new = np.concatenate([np.atleast_2d(X_init[:, j] <= thresh).T for j, thresh in self.continuous_thresholds], axis=1).astype(float)
            X_new = np.c_[X_new, X_continuous_new]
        
        
        
        if negation and len(self.continuous_thresholds) > 0:
            X_neg = np.concatenate([np.atleast_2d(X_init[:, j] > thresh).T for j, thresh in self.continuous_thresholds], axis=1).astype(float)
            X_new = np.c_[X_new, X_neg]
            feature_names_out += [f'{j} > {thresh}' for j, thresh in self.continuous_thresholds]

        return pd.DataFrame(X_new, columns=feature_names_out, dtype=int)
        