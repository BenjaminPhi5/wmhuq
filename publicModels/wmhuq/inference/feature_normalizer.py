from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, k=95, cut=True, log_features=None, return_arr=True):
        self.k = k
        self.cut = cut
        self.log_features = log_features
        self.means_ = {}
        self.stds_ = {}
        self.k_thresholds_ = {}
        self.return_arr = return_arr
        
    def load_weights(self, weight_file):
        weights = np.load(weight_file, allow_pickle=True)
        self.log_features = weights['log_features']
        columns = weights['columns']
        k_thresholds = weights['k_thresholds']
        means = weights['means']
        stds = weights['stds']
        
        self.k_thresholds_ = dict(zip(columns, k_thresholds))
        self.means_ = dict(zip(columns, means)) 
        self.stds_ = dict(zip(columns, stds))
        
    def fit(self, X, y=None):
        X = X.copy()
        
        for feature in X.columns:
            # optionally Log transform the feature
            if self.log_features is not None and feature in self.log_features:
                X[feature] = np.log(X[feature] + 1)
            
            # Calculate the k% threshold
            k_threshold = np.percentile(X[feature], self.k)
            self.k_thresholds_[feature] = k_threshold
            
            # Select the bottom k% of the data
            k_percent_data = X[X[feature] <= k_threshold][feature]
            
            # Calculate mean and std using the bottom k% of the data
            self.means_[feature] = k_percent_data.mean()
            self.stds_[feature] = k_percent_data.std()
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for feature in X.columns:
            # Log transform the feature
            if self.log_features is not None and feature in self.log_features:
                X[feature] = np.log(X[feature] + 1)
            
            # Clip the feature if cut is True
            if self.cut:
                X[feature] = np.clip(X[feature], None, self.k_thresholds_[feature])
            
            # Z-score normalization using parameters from the bottom k% of the data
            X[feature] = (X[feature] - self.means_[feature]) / self.stds_[feature]

        if self.return_arr:
            return X.values
        
        return X
