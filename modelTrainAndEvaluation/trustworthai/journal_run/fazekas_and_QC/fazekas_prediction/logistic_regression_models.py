from sklearn.linear_model import LogisticRegression
from trustworthai.utils.fitting_and_inference.optimizer_constructor_v2 import OptimizerConfigurator
import torch
import torch.nn as nn
import math
from trustworthai.journal_run.new_MIA_fazekas_and_QC.general_utils import VPrint, get_Xy_sets, get_fold, Rescaler
import numpy as np

class LogisticRegressionModel(nn.Module):
    def __init__(self, features, classes, max_iter=5000, balanced=True, alpha=0.05, beta=0.5, tol=0.0001, n_iter_no_change=None, verbose=False, device='cpu', weight_multiplier=None):
        super().__init__()
        self.layer = nn.Linear(features, classes, bias=True)
        optimizer_configurator = OptimizerConfigurator("Adam lr:2e-2 weight_decay:0")
        self.optimizer = optimizer_configurator(self.parameters())
        self.max_epochs=max_iter
        self.alpha = alpha
        self.beta = beta
        self.balanced = balanced
        self.device = device
        self.weight_multiplier = weight_multiplier
        
        # stopping criteria
        self.tol = tol
        if n_iter_no_change == None:
            n_iter_no_change = int(self.max_epochs/10)
        self.n_iter_no_change = n_iter_no_change
        
        self.trained = False
        self.verbose = verbose
        self.vprint = VPrint(verbose)
        
        generator = torch.Generator().manual_seed(42)
        nn.init.normal_(self.layer.weight, mean=0.0, std=0.5/(math.sqrt(features + classes)), generator=generator)
        nn.init.normal_(self.layer.bias, mean=0.0, std=0.5/(math.sqrt(classes)), generator=generator)
        
        self.layer = self.layer.to(self.device)
        
    def forward(self, x):
        return self.layer(x)
    
    def configure_criterion(self, y):
        if not self.balanced:
            self.criterion = nn.CrossEntropyLoss()
            return 
        
        # Calculate class weights
        class_counts = torch.bincount(y)
        self.vprint(class_counts)
        class_weights = 1. / class_counts.float()  # Inverse of class counts
        class_weights /= class_weights.min()       # Normalize to make the least frequent class have weight 1.0
        
        if self.weight_multiplier:
            class_weights *= torch.Tensor(self.weight_multiplier)
        
        self.class_weights = class_weights.to(self.device)

        # Create an instance of the CrossEntropyLoss with class weights
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
    
    def fit(self, X, y, X_val=None, y_val=None):
        if self.trained:
            raise Exception("this model has already been trained. Please initialize a new model")
        
        self.column_order = X.columns
        
        X = torch.from_numpy(X.values.astype(np.float32))
        y = torch.from_numpy(y).type(torch.long)
        
        if X_val is not None:
            X_val = torch.from_numpy(X_val.values.astype(np.float32))
            y_val = torch.from_numpy(y_val).type(torch.long)
        
        self.configure_criterion(y)
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        best_loss = torch.inf
        epochs_no_improvement = 0
        
        self.vprint("converge at: ", self.n_iter_no_change)
        
        for epoch in range(self.max_epochs):
            if epochs_no_improvement >= self.n_iter_no_change:
                self.vprint("converged")
                break
            
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self(X)

            # Compute and print loss
            class_loss = self.criterion(y_pred.squeeze(), y)
            loss = class_loss + self.layer.weight.abs().sum() * self.alpha * self.beta + self.layer.weight.abs().square().sum() * self.alpha * (1 - self.beta)

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if class_loss < best_loss:
                best_loss = class_loss
                epochs_no_improvement = 0
            elif class_loss < best_loss + self.tol:
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1
            
            if self.verbose and X_val is not None and epoch % 1000 == 0:
                with torch.no_grad():
                    y_val_pred = self(X_val)
                    val_class_loss = self.criterion(y_val_pred.squeeze(), y_val)
                    self.vprint(f"epoch: {epoch} loss: {class_loss.item():.2f} val loss: {val_class_loss.item():.2f}, no_change:{epochs_no_improvement}")
            
        self.trained = True
            
    def predict(self, X):
        assert (self.column_order == X.columns).all() # columns must be in same order for prediction to work correctly.
        with torch.no_grad():
            X = torch.from_numpy(X.values.astype(np.float32)).to(self.device)
            y_pred = self(X).argmax(dim=1).cpu().numpy()
        return y_pred

                                

def run_prediction_sklearn(df, X_fields, y_field, verbose=False, n_splits=5, val_proportion=0.0, stratify_target=None, rescale=True, penalty='l1', tol=0.1, C=0.15,max_iter=100000, solver='saga'):
    
    df = df.copy() # just to make sure I don't accidently modify the original at any point.
    
    X, y = get_Xy_sets(df, X_fields, y_field)
    
    X['seg_volume_pred_0.45'] = np.log(X['seg_volume_pred_0.45'].values)
    
    all_results = [[], [], []]
    for fold_num in (range(n_splits)):
        (X_train, y_train), (X_test, y_test), (X_val, y_val) = get_fold(X, y, fold_num=fold_num, n_splits=n_splits, val_proportion=val_proportion, stratify_target=stratify_target)
        if rescale:
            rescaler = Rescaler()
            rescaler.fit(X_train)
            X_train = rescaler.predict(X_train)
            X_test = rescaler.predict(X_test)
            X_val = rescaler.predict(X_val)

        # define model and train it
        clf = LogisticRegression(penalty=penalty, dual=False, tol=tol, C=C, fit_intercept=True, intercept_scaling=1, class_weight='balanced', 
                         random_state=None, solver=solver, max_iter=max_iter, multi_class='multinomial', verbose=0, warm_start=False,
                         n_jobs=6, l1_ratio=None)
        clf.fit(X_train, y_train)

        y_hat_train, y_hat_val, y_hat_test = clf.predict(X_train), clf.predict(X_val), clf.predict(X_test)
        
        all_results[0].append((y_train, y_hat_train))
        all_results[1].append((y_val, y_hat_val))
        all_results[2].append((y_test, y_hat_test))
     
    # combine the results for all the different splits together
    for idx in range(3):
        all_results[idx] = (np.concatenate([a[0] for a in all_results[idx]]), np.concatenate([a[1] for a in all_results[idx]]))
        
    return all_results
        