"""
Receives one checkpoint file of the trained model generated during training/validation
Creates the list of test samples for that fold
Excludes test slices that do not conform with the criteria used for defining the training and validation samples 
Runs inference with the loaded model
Output average Sorensen Dice for the run 
"""

import sys,os,glob,pickle,torch
import multiprocessing as mp
from torch.cuda.amp import autocast
from natsort import natsorted
import numpy as np
from mainLoop import createFolds,createListOfSamples
from pytorchClasses import Dataset
from pytorch_util import diceLoss,initModel

if __name__ == '__main__':
    
    modelFile = sys.argv[1]
    
    foldIdx = (os.path.basename(os.path.normpath(modelFile))[:-4]).split('_')[0][-1]
    foldIdx = int(foldIdx)
    
    inferenceParams = {'batch_size': 1,'shuffle': True,'num_workers': int(0.8*mp.cpu_count())}
    multiGPU = False

    projectDir = '/nfs/home/fcunha.it/deepLearning/challenges/LUNA16'    
    subsets = natsorted(glob.glob(os.path.join(projectDir,'subset*')))

    _,_,testSet = createFolds(foldIdx,subsets)

    # read the dictionary that encodes the UIDs and respective slices with annotations
    with open('/nfs/home/fcunha.it/deepLearning/challenges/LUNA16/maskedSlices.pkl', 'rb') as f:
        maskedSlices = pickle.load(f)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    segmentationModel = initModel(device,useMultiGPUs=False,useVolContext=True)
    segmentationModel.load_state_dict(torch.load(modelFile))
    segmentationModel.eval()
    
    
    testSamples = createListOfSamples(testSet,maskedSlices,filterByAnnotationsMask=True)
    testSet = Dataset(testSamples)
    testing_generator = torch.utils.data.DataLoader(testSet, **inferenceParams)

    print('----------------------Running Inference----------------------')
    testBatchLosses = []
    with torch.set_grad_enabled(False): # Performs inference on validation data and prevents back-propagation
        for slices_batch, masks_batch in testing_generator:
            #Transfer to GPU
            slices_batch = slices_batch.to(device)
            masks_batch = masks_batch.to(device)
            
            with autocast():
                # Model computations
                predictedSegmentation = segmentationModel(slices_batch) #predicted masks for the batch

                testLoss = diceLoss((predictedSegmentation,masks_batch))
                # mean of the batch
                testLoss = testLoss.mean()
                # append to the list of batch losses
            testBatchLosses.append(testLoss.cpu().data.numpy())
    
    # compute the mean of all batches
    diceCoeffs = [1 - x for x in testBatchLosses]
    globalTestDice = np.mean(diceCoeffs)
    globalTestDiceStd = np.std(diceCoeffs)
    print('\nAverage Sorensen-Dice for fold %d:  %.4f+-%.4f' % (foldIdx,globalTestDice,globalTestDiceStd))
    print('\nFinished!')
