"""
code modified from the wmh challenge metrics github

I should put their copywrite notice here I think

"""

import SimpleITK as sitk
import numpy as np
import torch
from trustworthai.utils.print_and_write_func import print_and_write
from trustworthai.utils.logits_to_preds import mle_batch
from tqdm import tqdm
import scipy

# def dice(y_pred, y_true):
#     if do_softmax:
#         y_pred = torch.nn.functional.softmax(y_pred, dim=1).argmax(dim=1)
#     else:
#         y_pred = y_pred.argmax(dim=1)
#     #print(y_pred.shape, y_true.shape)
#     denominator = torch.sum(y_pred) + torch.sum(y_true)
#     numerator = 2. * torch.sum(torch.logical_and(y_pred, y_true))
#     return numerator / denominator

# def AVD(y_pred, y_true):
#     y_pred = y_pred.argmax(dim=1)
#     avd = (y_pred.sum() - y_true.sum()).abs() / y_true.sum() * 100
#     return avd.item()

def dice(pred, target):
    pred = mle_batch(pred, dtype=torch.long)
    target = target.type(torch.long)
    return getDSC(target, pred)

def AVD(pred, target):
    pred = mle_batch(pred, dtype=torch.long)
    target = target.type(torch.long)
    return getAVD(target, pred)

def avd_mean_analysis(results_text_file, means3d, ys3d):
    avds_mean = []
    for ind in range(len(ys3d)):
        mu = means3d[ind]
        target = ys3d[ind]
        avds_mean.append(AVD(mu, target))

    print_and_write(results_text_file, f"mean AVD", newline=1)
    print_and_write(results_text_file, torch.Tensor(avds_mean).mean())
    
def dice_mean_analysis(results_text_file, means3d, ys3d):
    # compute dice for the mean produced by the model
    dices_mean = []
    for ind in range(len(ys3d)):
        mu = means3d[ind]
        target = ys3d[ind]
        dices_mean.append(dice(mu, target))

    dices_mean = torch.Tensor(dices_mean)
    print_and_write(results_text_file, f"mean dice", newline=1)
    print_and_write(results_text_file, dices_mean.mean())
    
    
def dice_across_samples_analysis(results_text_file, ys3d, samples3d):
    # compute the dice per sample, per individual
    # TODO: check whether this code is actually correct, like is it not
    # [s][ind] not [ind][s]
    dices3d = []
    for ind in tqdm(range(len(samples3d)), position=0, leave=True, ncols=150):
        sample_dices = []
        for s in range(len(samples3d[ind])):
            y_hat = samples3d[ind][s]
            y = ys3d[ind]
            sample_dices.append(dice(y_hat, y))
        dices3d.append(sample_dices)

    tensor_alldice3d = torch.stack([torch.Tensor(ds) for ds in dices3d], dim=0).swapaxes(0,1)

    # best dice mean. This is a little dissapointing.
    bdm = tensor_alldice3d.max(dim=0)[0].mean()
    print_and_write(results_text_file, f"best_dice_mean", newline=1)
    print_and_write(results_text_file, bdm)
    
    return tensor_alldice3d

def getDSC(testImage, resultImage):    
        """Compute the Dice Similarity Coefficient."""
        # testArray   = sitk.GetArrayFromImage(testImage).flatten()
        # resultArray = sitk.GetArrayFromImage(resultImage).flatten()
        testArray = testImage.view(-1).cpu().numpy()
        resultArray = resultImage.view(-1).cpu().numpy()

        # similarity = 1.0 - dissimilarity
        return 1.0 - scipy.spatial.distance.dice(testArray, resultArray) 


def getHausdorff(testImage, resultImage):
    """Compute the Hausdorff distance."""

    testImage = sitk.GetImageFromArray(testImage)
    resultImage = sitk.GetImageFromArray(resultImage)

    # Hausdorff distance is only defined when something is detected
    resultStatistics = sitk.StatisticsImageFilter()
    resultStatistics.Execute(resultImage)
    if resultStatistics.GetSum() == 0:
        return float('nan')

    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    eTestImage   = sitk.BinaryErode(testImage, (1,1,0) )
    eResultImage = sitk.BinaryErode(resultImage, (1,1,0) )

    hTestImage   = sitk.Subtract(testImage, eTestImage)
    hResultImage = sitk.Subtract(resultImage, eResultImage)    

    hTestArray   = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)   

    # Convert voxel location to world coordinates. Use the coordinate system of the test image
    # np.nonzero   = elements of the boundary in numpy order (zyx)
    # np.flipud    = elements in xyz order
    # np.transpose = create tuples (x,y,z)
    # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
    testCoordinates   = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hTestArray) ))]
    resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hResultArray) ))]


    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):    
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]

    # Compute distances from test to result; and result to test
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)    

    return max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))


def getLesionDetection(testImage, resultImage):    
    """Lesion detection metrics, both recall and F1."""
    testImage = sitk.GetImageFromArray(testImage)
    resultImage = sitk.GetImageFromArray(resultImage)

    # Connected components will give the background label 0, so subtract 1 from all results
    ccFilter = sitk.ConnectedComponentImageFilter()    
    ccFilter.SetFullyConnected(True)

    # Connected components on the test image, to determine the number of true WMH.
    # And to get the overlap between detected voxels and true WMH
    ccTest = ccFilter.Execute(testImage)    
    lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))

    ccTestArray = sitk.GetArrayFromImage(ccTest)
    lResultArray = sitk.GetArrayFromImage(lResult)

    # recall = (number of detected WMH) / (number of true WMH) 
    nWMH = len(np.unique(ccTestArray)) - 1
    if nWMH == 0:
        recall = 1.0
    else:
        recall = float(len(np.unique(lResultArray)) - 1) / nWMH

    # Connected components of results, to determine number of detected lesions
    ccResult = ccFilter.Execute(resultImage)
    lTest = sitk.Multiply(ccResult, sitk.Cast(testImage, sitk.sitkUInt32))

    ccResultArray = sitk.GetArrayFromImage(ccResult)
    lTestArray = sitk.GetArrayFromImage(lTest)

    # precision = (number of detections that intersect with WMH) / (number of all detections)
    nDetections = len(np.unique(ccResultArray)) - 1
    if nDetections == 0:
        precision = 1.0
    else:
        precision = float(len(np.unique(lTestArray)) - 1) / nDetections

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return recall, f1    


def getAVD(testImage, resultImage):   
    """Volume statistics."""
    testImage = sitk.GetImageFromArray(testImage)
    resultImage = sitk.GetImageFromArray(resultImage)

    # Compute statistics of both images
    testStatistics   = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()

    testStatistics.Execute(testImage)
    resultStatistics.Execute(resultImage)

    return float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100

def do_challenge_metrics(testImage, resultImage, print_results=False):
    """Main function"""
    dsc = getDSC(testImage, resultImage)
    try:
        h95 = getHausdorff(testImage, resultImage)
    except:
        h95 = 100
    avd = getAVD(testImage, resultImage)    
    recall, f1 = getLesionDetection(testImage, resultImage)    

    if print_results:
        print('Dice',                dsc,       '(higher is better, max=1)')
        print('HD',                  h95, 'mm',  '(lower is better, min=0)')
        print('AVD',                 avd,  '%',  '(lower is better, min=0)')
        print('Lesion detection', recall,       '(higher is better, max=1)')
        print('Lesion F1',            f1,       '(higher is better, max=1)')

    return dsc, h95, avd, recall, f1

def per_model_chal_stats(preds3d, ys3d):
    stats = []
    for i in tqdm(range(len(ys3d)), position=0, leave=True):
        ind_stats = do_challenge_metrics(ys3d[i].type(torch.long), preds3d[i].argmax(dim=1).type(torch.long))
        stats.append(ind_stats)

    tstats = torch.Tensor(stats)
    dices = tstats[:,0]
    hd95s = tstats[:,1]
    avds = tstats[:,2]
    recalls = tstats[:,3]
    f1s = tstats[:,4]

    data = {"dice":dices, "hd95":hd95s, "avd":avds, "recall":recalls, "f1":f1s}

    return data

def write_model_metric_results(results_file, data):
    print_and_write(results_file, f"dice", newline=1)
    print_and_write(results_file, data["dice"],newline=1)
    print_and_write(results_file, f"hd95",newline=1)
    print_and_write(results_file, data["hd95"],newline=1)
    print_and_write(results_file, f"avd",newline=1)
    print_and_write(results_file, data["avd"],newline=1)
    print_and_write(results_file, f"recall",newline=1)
    print_and_write(results_file, data["recall"],newline=1)
    print_and_write(results_file, f"f1",newline=1)
    print_and_write(results_file, data["f1"])