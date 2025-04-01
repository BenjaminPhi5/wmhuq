import torch
import pandas as pd
import numpy as np
import proplot as pplt
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize

def cleanup_df(df):
    ### removing columns that have Unnamed in the name
    unnamed_columns = [c for c in df.columns if "unnamed" in c.lower()]
    df.drop(columns=unnamed_columns, inplace=True)
    
    ### removing keys that have zero values
    number_columns = [key for key in df.columns if df[key].values.dtype != 'O']
    # print(number_columns)
    number_df = df[number_columns]
    zerosums = np.sum(number_df, axis=0) == 0
    zero_columns = zerosums.index[zerosums.values]
    df.drop(columns=zero_columns, inplace=True)
    
    ### removing participants that have zeros in the name
    number_columns = [key for key in df.columns if df[key].values.dtype != 'O']
    for key in number_columns:
        row_values = df[key].values
        num_zeros = (row_values == 0).sum()
        if num_zeros == 0:
            continue
        if len(np.unique(row_values)) < 20: # don't filter out one hot encoded columns
            continue
        # if num_zeros > 0 and num_zeros < 4:
        #     indexes = df[df[key] == 0].index
        #     df.drop(indexes, inplace=True, axis=0)
        if num_zeros > 0 and num_zeros >=4:
            df.drop(columns=[key], inplace=True)

def remove_nans(dfc, verbose=False):
    nan_rows = dfc[dfc.isnull().any(axis=1)]
    nan_cols = dfc.columns[dfc.isnull().any()]
    if verbose:
        print("nans removed: ", len(nan_rows))
    dfc = dfc.drop(labels=nan_rows.index.values, axis=0)
    return dfc

def get_Xy_sets_exact(df, X_fields, y_field, include_ID=False):
    X_keys = []
    if include_ID:
        X_keys.append("ID")
    
    for key in df.keys():
        for field in X_fields:
            if field == key:
                X_keys.append(key)
                break
                
    if y_field in X_keys:
        X_keys.remove(y_field)
        
    selected_keys = X_keys + [y_field]
    
    df = df[selected_keys]
    
    # print(df.shape)
    df = remove_nans(df)
    # print(df.shape)
    
    X_keys = sorted(X_keys) # make sure that the features are always presented in the same order
    
    X, y = df[X_keys], df[y_field]
    y = y.values.squeeze()
    
    return X, y

def get_Xy_sets(df, X_fields, y_field, include_ID=False):
    X_keys = []
    if include_ID:
        X_keys.append("ID")
    
    for key in df.keys():
        for field in X_fields:
            if field in key:
                X_keys.append(key)
                
    selected_keys = X_keys + [y_field]
    
    df = df[selected_keys]
    
    # print(df.shape)
    df = remove_nans(df)
    # print(df.shape)
    
    X, y = df[X_keys], df[y_field]
    y = y.values.squeeze()
    
    return X, y

def feature_key_match(key, feature_set):
    for f in feature_set:
        if f in key:
            return True
    return False

# old method from previous versions. The above is now used.
# def extract_Xy(df, label_class, verbose=False, kept_column=None, label_categories=None):
#     df = remove_nans(df, verbose)
#     if label_categories == None:
#         label_categories = ['WMH_PV', 'WMH_Deep', 'Total', 'totatal_fazekas']
#     if kept_column != None:
#         y_reg = df[kept_column].values
#     y = df[label_class].values
#     X = df.drop(columns=label_categories)
#     if label_class not in label_categories:
#         X = X.drop(columns=label_class)
#     if kept_column != None:
#         X[kept_column] = y_reg
#     return X, y

def shuffle(df, random_state=42):
    df = df.copy()
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df

class VPrint():
    def __init__(self, verbose=True):
        self.verbose = verbose
    def __call__(self, *prompts):
        if self.verbose:
            print(*prompts)
        # else do nothing.

        
from sklearn.preprocessing import (
    Normalizer,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    PowerTransformer,
    QuantileTransformer,
    SplineTransformer
)
    

class Rescaler2():
    def fit(self, X):
        keys = sorted(X.columns.values)
        X = X[keys]
        norm_keys = []
        for key in X.keys():
            values = X[key].values
            if len(np.unique(values)) > 5:
                norm_keys.append(key)
        self.norm_keys = norm_keys
        X_keys = X[norm_keys]
        # norm = StandardScaler() # eh not identical as mine somehow. possibly due to ddof?
        # norm = Normalizer(norm='l2') # terrible
        # norm = MaxAbsScaler() # terrible
        # norm = MinMaxScaler() # eh. similar accuracy on test. struggles on cvd data somehow
        # norm = RobustScaler() # eh
        # norm = PowerTransformer()
        norm = QuantileTransformer(n_quantiles=50, output_distribution='normal')
        norm.fit(X_keys)
        self.norm = norm
    def predict(self, X):
        keys = sorted(X.columns.values)
        X = X[keys]
        X = X.copy()
        X[self.norm_keys] = self.norm.transform(X[self.norm_keys])
        
        return X
        
class Rescaler():
    def fit(self, X):
        norm_parameters = []
        for key in X.keys():
            values = X[key].values
            if len(np.unique(values)) > 5:
                mean = np.mean(values)
                std = np.std(values)
                norm_parameters.append((key, mean, std))
        self.norm_parameters = norm_parameters
    def predict(self, X):
        X = X.copy()
        for key, mean, std in self.norm_parameters:
            X[key] = (X[key] - mean) / std
        
        return X


def plot_confusion_matrix(y_true, y_hat, fig=None, ax=None, cmap="nuuk", class_labels=None):
    # Calculate N x N confusion matrix
    cm = confusion_matrix(y_true, y_hat)

    # remove any case where a class doesn't exist in the ground truth, but is predicted by the model
    # existing_classes = np.where(cm.sum(axis=1)>0)[0]
    if class_labels is not None:
        target_tick_labels = np.array(class_labels)
    else:
        target_tick_labels = np.arange(cm.shape[0])
    # cm = cm[existing_classes]
    # target_tick_labels = target_tick_labels[existing_classes]
    
    # Normalize confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # for the colourmap, i like the buda, tokyo, lajolla and nuuk colourmaps
    
    # Plot confusion matrix
    if fig == None:
        fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1, rasterized=True)
    
    ax.set(
        yticks=np.arange(cm.shape[0]),
        xticks=np.arange(cm.shape[1]),
        xticklabels=class_labels if class_labels is not None else np.arange(cm.shape[1]),
        yticklabels=target_tick_labels,
        ylabel='True label',
        xlabel='Predicted label'
    )

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' 
    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] < thresh else "black", fontsize=13)
            
def confusion_matrices(results_sets, cmap="nuuk", class_labels=None, ax_names=None):
    fig, axs = pplt.subplots(refwidth="20em", ncols=len(results_sets), nrows=1)
    if ax_names is None:
        ax_names = ['' for _ in range(len(axs))]
    for (true, predicted), ax in zip(results_sets, axs):
        plot_confusion_matrix(true, predicted, fig, ax, cmap, class_labels)
        
    for ax, title in zip(axs, ax_names):
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')

        ax.tick_params(
            axis='both',          # changes apply to both x and y axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            labelbottom=True) # labels along the b
        
        ax.set_title(title)

    axs.format(grid=False, fontsize=15) # grid off for all axes
    
def generate_classification_stats(results, print_results=True, return_stats=True):
    vprint = VPrint(print_results)
    
    # Example labels and predictions
    y_true = results[0]
    y_pred = results[1]
    # print("evaluation set size: ", len(y_true)) 

    # Number of classes
    n_classes = len(np.unique(y_true))

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    vprint(f"Accuracy: {accuracy:.3f}")

    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    vprint(f"Balanced Accuracy: {balanced_accuracy:.3f}")

    # Precision, Recall, and F1-Score
    class_report = classification_report(y_true, y_pred)
    vprint("Classification Report:\n", class_report)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    vprint("Confusion Matrix:\n", conf_matrix)

    # ROC-AUC
    # Note: ROC-AUC can be computed for each class in a one-vs-all manner in a multiclass setting
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    y_pred_binarized = label_binarize(y_pred, classes=np.unique(y_true))
    roc_auc = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr')
    vprint(f"ROC AUC Score: {roc_auc}")
    
    # f1
    
    
    if not return_stats:
        return
    return {
        "Accuracy":accuracy,
        "Balanced Accuracy":balanced_accuracy,
        "Classification Report":class_report,
        "Confusion Matrix": conf_matrix,
        "ROC AUC Score": roc_auc,
        "f1_weighted":f1_score(y_true, y_pred, average="weighted"),
        "f1_macro":f1_score(y_true, y_pred, average="macro"),
        "f1_micro":f1_score(y_true, y_pred, average="micro")
        
    }
    
def shift_and_log_features(df, feature_matches, verbose=False):
    df = df.copy()
    for key in df.keys():
        for f in feature_matches:
            if f in key:
                if verbose:
                    print(f"taking log of {key}")
                values = df[key].values
                df[key] = np.log(values - min(values) + 0.01)
                continue
    return df

def get_fold(X, y, fold_num, n_splits=5, val_proportion=0.2, stratify_target=None):
    
    X = X.reset_index(drop=True)
    
    kf = StratifiedKFold(n_splits=n_splits)
    if stratify_target == None:
        stratify_values = y
    else:
        stratify_values = X[stratify_target]

    for i, (train_idx, test_idx) in enumerate(kf.split(X, stratify_values)):
        if i == fold_num:
            train_X = X.iloc[train_idx]
            train_y = y[train_idx]
            test_X = X.iloc[test_idx]
            test_y = y[test_idx]
            break
        
    train_X = train_X.reset_index(drop=True)

    if val_proportion > 0:
        kf_val = StratifiedKFold(n_splits=int(1//val_proportion))
        if stratify_target == None:
            stratify_values = train_y
        else:
            stratify_values = train_X[stratify_target]

        for i, (train_idx, val_idx) in enumerate(kf_val.split(train_X, stratify_values)):
            if i == 0:
                val_X = train_X.iloc[val_idx]
                val_y = train_y[val_idx]
                train_X = train_X.iloc[train_idx]
                train_y = train_y[train_idx]

                break

        if stratify_target != None:
            train_X = train_X.drop(columns=stratify_target)
            val_X = val_X.drop(columns=stratify_target)
            test_X = test_X.drop(columns=stratify_target)

        return (train_X, train_y), (test_X, test_y), (val_X, val_y)
    
    else:
        if stratify_target != None:
            train_X = train_X.drop(columns=stratify_target)
            test_X = test_X.drop(columns=stratify_target)

        return (train_X, train_y), (test_X, test_y), (test_X, test_y) # just return the test set as a dummy val in the val prop = 0 case.
    
def create_minimal_correlation_keys(df, ignore_keys=None, threshold=0.8, verbose=False):
    # we ignore the target keys when computing correlation and deciding which features to remove
    
    if ignore_keys != None:
        df = df.drop(columns=ignore_keys)
    
    remaining_keys = list(df.keys())
    current_keys = [remaining_keys[0]]
    
    for key in remaining_keys[1:]:
        cols = df[current_keys + [key]]
        corr = cols.corr()
        corr_values = abs(corr[key].values[:-1])
        corr_values[np.isnan(corr_values)] = 0 # nan correlation values are ignored (they will register as argmax so we set them to zero).
        # print(corr_values)
        try:
            max_corr = max(corr_values)
        except:
            print("key failed: ", key, corr_values)
        # if max_corr == 0:
        #     print(key)
        if max_corr > threshold:
            if verbose:
                print(f"excluding {key}, max corr of {max_corr} with {list(corr.keys())[np.argmax(corr_values)]}")
            #print(corr[key].values)
            continue
        else:
            current_keys.append(key)
    
    if verbose:
        print(f"retained keys = {len(current_keys)} (total {len(df.keys())})")
        
    return current_keys
