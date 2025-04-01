import torch
# def precision()

# def recall()

# def pr_curve()

# def apr_curve()

def specificity(y_pred, y_true):
    # TN / (TN + FP)
    true_negative = torch.sum((1.-y_pred) * (1.-y_true))
    false_positive = torch.sum(y_pred * (1-y_true))
    
    return true_negative / (true_negative + false_positive)

def sensitivity(y_pred, y_true):
    # TP / (TP + FN)
    true_positive = torch.sum(y_pred * y_true)
    false_negative = torch.sum((1.-y_pred) * y_true)
    
    return true_positive / (true_positive + false_negative)

def f2(y_pred, y_true):
    # see tversky loss paper, commonly used where recall more helpful than precision
    # f2 = 5TP / (5TP + 4FN + FP)
    true_positive = torch.sum(y_pred * y_true)
    false_negative = torch.sum((1.-y_pred) * y_true)
    false_positive = torch.sum(y_pred * (1-y_true))
    
    numerator = 5. * true_positive
    denominator = (
                numerator 
                + 4. * false_negative 
                + false_positive
    )
    
    return numerator / denominator

# def f1()

# TODO: complete.
def f_beta():
    pass # todo see machinelearningmastery post on this!


def IOU(y_pred, y_true):
    # this is 1 - standard jaccard (p=1)
    true_positive = torch.sum(y_pred * y_true)
    return true_positive / (torch.sum(y_pred) + torch.sum(y_true) - true_positive)
    