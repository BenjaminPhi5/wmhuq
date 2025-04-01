import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from wmhuq.inference import CustomNormalizer
import os
import copy
import pandas as pd

K = 95
CUT = True

def load_logreg_(weights_file):
    weights = np.load(weights_file, allow_pickle=True)
    coef = weights['coef']
    intercept = weights['intercept']
    classes = weights['classes']
    columns = weights['cols']
    
    model = LogisticRegression()
    model.coef_ = coef
    model.intercept_ = intercept
    model.classes_ = classes
    
    return model, columns
        

class ColumnFilter(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def transform(self, X):
        return X[self.columns]
    

def load_logreg_model(norm_weight_path, model_weight_path):
    norm = CustomNormalizer(k=K, cut=CUT)
    norm.load_weights(norm_weight_path)
    clf, columns = load_logreg_(model_weight_path)
    
    return Pipeline([("colfilter", ColumnFilter(columns)), ("norm", norm), ("clf", clf)])
    
class FazekasModel():
    def __init__(self, dwmh_model, pvwmh_model):
        self.dwmh_model = dwmh_model
        self.pvwmh_model = pvwmh_model

        dwmh_normer = copy.deepcopy(self.dwmh_model["norm"])
        dwmh_normer.cut = False
        dwmh_normer.return_arr = False
        pvwmh_normer = copy.deepcopy(self.pvwmh_model["norm"])
        pvwmh_normer.cut = False
        pvwmh_normer.return_arr = False
        self.dwmh_features = Pipeline([("colfilter", self.dwmh_model["colfilter"]), ("norm", dwmh_normer)])
        self.pvwmh_features = Pipeline([("colfilter", self.pvwmh_model["colfilter"]), ("norm", pvwmh_normer)])
        self.copy_cols = [c for c in self.pvwmh_model["colfilter"].columns if c not in self.dwmh_model["colfilter"].columns]
    
    def __call__(self, X):
        dwmh_results = self.dwmh_model.predict_proba(X)
        pvwmh_results =  self.pvwmh_model.predict_proba(X)

        dwmh_features = self.dwmh_features.transform(X)
        pvwmh_features = self.pvwmh_features.transform(X)
        combined_features = dwmh_features.copy()
        combined_features[self.copy_cols] = pvwmh_features[self.copy_cols]
        
        probs = {
            "DWMH_0":[dwmh_results[0, 0]],
            "DWMH_1":[dwmh_results[0, 1]],
            "DWMH_2":[dwmh_results[0, 2]],
            "DWMH_3":[dwmh_results[0, 3]],
            "PVWMH_0":[pvwmh_results[0, 0]],
            "PVWMH_1":[pvwmh_results[0, 1]],
            "PVWMH_2":[pvwmh_results[0, 2]],
            "PVWMH_3":[pvwmh_results[0, 3]],
        }
        
        return pd.DataFrame(probs), combined_features
    
def load_fazekas_model(weights_folder):
    dwmh_norm = os.path.join(weights_folder, "dwmh_scaler_weights.npz")
    dwmh_weight = os.path.join(weights_folder, "dwmh_fazekas_logreg_model_weights.npz")
    
    dwmh_model = load_logreg_model(dwmh_norm, dwmh_weight)
    
    pvwmh_norm = os.path.join(weights_folder, "pvwmh_scaler_weights.npz")
    pvwmh_weight = os.path.join(weights_folder, "pvwmh_fazekas_logreg_model_weights.npz")
    
    pvwmh_model = load_logreg_model(pvwmh_norm, pvwmh_weight)
    
    fazekas_model = FazekasModel(dwmh_model, pvwmh_model)
    return fazekas_model
    
class QCModel():
    def __init__(self, qc_base_model):
        self.model = qc_base_model
        
    def __call__(self, X):
        # X = X.reshape(1, -1)
        return pd.DataFrame({"QC_Score":[self.model.predict_proba(X)[0, 0]]})

def load_qc_model(weights_folder):
    qc_norm = os.path.join(weights_folder, "qc_scaler_weights.npz")
    qc_weight = os.path.join(weights_folder, "qc_logreg_model_weights.npz")
    
    qc_model = QCModel(load_logreg_model(qc_norm, qc_weight))
    return qc_model
