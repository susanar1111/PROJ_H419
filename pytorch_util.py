import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

from pytorchClasses import CustomUNet

def initModel(cudaDevice,useMultiGPUs=1,useVolContext=False):

    torch.cuda.empty_cache()
    if useVolContext:
        segmentation_model = CustomUNet(in_channels=3,n_classes=1,depth=4,wf=5,padding=True,batch_norm=True,up_mode='upconv')
    else:
        segmentation_model = CustomUNet(in_channels=1,n_classes=1,depth=4,wf=5,padding=True,batch_norm=True,up_mode='upconv')
    
    if useMultiGPUs:
        if torch.cuda.device_count() > 1:
            segmentation_model = nn.DataParallel(segmentation_model)
    
    segmentation_model = segmentation_model.to(cudaDevice)
    
    return segmentation_model
    
def initOptimizer(modelToTrain):
    return Adam(modelToTrain.parameters(),lr=1e-4)

def diceLoss(args):
    # Soft-dice (multiply the sigmoid activations directly without thresholding => otherwise, optimize the gradients would be )
    
    prediction, groundTruth = args
    eps=1
    
    #prediction = torch.reshape(prediction,groundTruth.shape)

    diceLabel_g = groundTruth.sum(dim=[1,2,3])
    dicePrediction_g = prediction.sum(dim=[1,2,3])

    diceCorrect_g = (prediction * groundTruth).sum(dim=[1,2,3])

    diceCoeff = (2 * diceCorrect_g + eps) /(dicePrediction_g + diceLabel_g + eps)

    return 1 - diceCoeff


def revertNormalization(srcImg,numBits=8):
    to_max = 2**numBits - 1
    to_min = 0
    
    from_max = np.amax(srcImg)
    from_min = np.amin(srcImg)
    
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((srcImg - from_min) / float(from_range), dtype=float)
    
    output = to_min + (scaled * to_range)
    
    if numBits == 8:
        output = np.uint8(output)
    elif numBits == 16:
        output = np.uint16(output)
    elif numBits == 1: # to visualize the masks
        output = 255*np.uint8(output)
    return output
