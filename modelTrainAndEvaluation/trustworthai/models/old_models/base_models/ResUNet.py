"""
based on https://github.com/assassint2017/MICCAI-LITS2017
which uses
I should really read all these to see what they are doing in this work, but for now I am copying the model to see if I can get a good baseline.


    Milletari F, Navab N, Ahmadi S A. V-net: Fully convolutional neural networks for volumetric medical image segmentation[C]//2016 Fourth International Conference on 3D Vision (3DV). IEEE, 2016: 565-571.
    Wong K C L, Moradi M, Tang H, et al. 3d segmentation with exponential logarithmic loss for highly unbalanced object sizes[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2018: 612-619.
    Yuan Y, Chao M, Lo Y C. Automatic skin lesion segmentation using deep fully convolutional networks with jaccard distance[J]. IEEE transactions on medical imaging, 2017, 36(9): 1876-1886.
    Salehi S S M, Erdogmus D, Gholipour A. Tversky loss function for image segmentation using 3D fully convolutional deep networks[C]//International Workshop on Machine Learning in Medical Imaging. Springer, Cham, 2017: 379-387.
    Brosch T, Yoo Y, Tang L Y W, et al. Deep convolutional encoder networks for multiple sclerosis lesion segmentation[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2015: 3-11.
    Xu W, Liu H, Wang X, et al. Liver Segmentation in CT based on ResUNet with 3D Probabilistic and Geometric Post Process[C]//2019 IEEE 4th International Conference on Signal and Image Processing (ICSIP). IEEE, 2019: 685-689.
    Krähenbühl P, Koltun V. Efficient inference in fully connected crfs with gaussian edge potentials[C]//Advances in neural information processing systems. 2011: 109-117.
"""

import torch
import torch.nn as nn
import torch.nn.fnunctional as F

class ResUNet(nn.Module):
    def __init__(self):
        super().__init__
        
        