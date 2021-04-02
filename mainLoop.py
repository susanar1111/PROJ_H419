from albumentations.augmentations.transforms import GridDistortion
from natsort import natsorted
import numpy as np
import pickle,os,glob,torch
import multiprocessing as mp
from pytorchClasses import Dataset
from torch.cuda.amp import GradScaler,autocast
import gc
from pytorch_util import initModel,initOptimizer,diceLoss,revertNormalization
from visdom import Visdom
import albumentations as A
from albumentations import Flip,GridDistortion,GaussianBlur
from albumentations.pytorch.transforms import ToTensorV2

def listFilesFrom(files,srcDirs,fileExtension): 
    
    if isinstance(srcDirs,str):
        srcDirs = [srcDirs]
        
    for subset_ in srcDirs:
        files_ = [os.path.join(subset_,x) for x in os.listdir(subset_) if x.endswith(fileExtension)]
        files.extend(files_)
    return files

def createFolds(foldIdx,dataFolders,N=10):
    # pdb.set_trace()
    trainingLowerIdx = foldIdx
    trainingUpperIdx = foldIdx + 8
        
    validationIdx = foldIdx+ 8
    testIdx = foldIdx + 9
    
    trainingIndices = np.take(np.arange(N),range(trainingLowerIdx,trainingUpperIdx),mode='wrap')
    validationIdx = np.take(np.arange(N),validationIdx,mode='wrap')
    testIdx = np.take(np.arange(N),testIdx,mode='wrap')
    
    trainingSubsets = [dataFolders[x] for x in trainingIndices]
    trainingSet = []
    listFilesFrom(trainingSet,trainingSubsets,'mhd')
    
    validationSubsets = dataFolders[validationIdx]
    validationSet = []
    listFilesFrom(validationSet,validationSubsets,'mhd')
    
    testingSubsets = dataFolders[testIdx]
    testingSet = []
    listFilesFrom(testingSet,testingSubsets,'mhd')
    
    return trainingSet,validationSet,testingSet

def createListOfSamples(candidateFiles,auxDict,filterByAnnotationsMask=False):

    listOfSamples = [] # initialize output variable
    
    # find the candidateFiles that are a key in auxDict => that have masks
    maskedCandidates = list(set(candidateFiles).intersection(set(auxDict.keys())))
    
    for candidateFile_ in maskedCandidates:
        # grab the corresponding subset
        subset_ = candidateFile_.split('/')[-2]
        uid_ = candidateFile_.split('/')[-1][:-4]
        
        # source folder to the data
        srcDir = os.path.join('/nfs/home/susana/deepLearning/challenges/LUNA16/data',subset_)
        masksDir = os.path.join(srcDir,'masks') 
        
        # take the masked slices only
        maskedSliceIdxs = auxDict[candidateFile_][-1]
        
        for sliceIdx_ in maskedSliceIdxs:
            # take the preprocessed slice name
            preprocessed_slice = os.path.join(srcDir,uid_+'_slice_'+str(sliceIdx_))
            preprocessed_slice+='.npy'
            # take the ground-truth slice mask
            slice_mask = os.path.join(masksDir,uid_+'_slice_'+str(sliceIdx_)+'_mask')
            slice_mask+='.npy'

            # Read the slice and the mask
            slice_mask_data = np.load(slice_mask)
            maskedPositions = np.where(slice_mask_data == 1)

            if filterByAnnotationsMask:
                # check if the masked positions are sufficiently centered on the image
                if all(maskedPositions[0] > 0.2*512) and all(maskedPositions[0] < 0.8*512) and all(maskedPositions[1] > 0.2*512) and all(maskedPositions[1] < 0.8*512):
                    # Proceed if there is a minimum number of masked pixels (discard too small annotation) 
                    if len(maskedPositions[0]) >= 250:
                        # verify the spatial "quality" of the source image (visual inspection of the data generation routine revealed a significant #slices with only texture at one lung )
                        listOfSamples.append((preprocessed_slice,slice_mask))
            else:
                listOfSamples.append((preprocessed_slice,slice_mask))


    return listOfSamples

if __name__== '__main__':
    
    vis = Visdom(port='8098')
   
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    trainingParams = {'batch_size': 14,'shuffle': True,'num_workers': int(0.8*mp.cpu_count())}
    multiGPU = False
    doAugmentation = True

    if doAugmentation: # create a composition of transforms
        train_transform = A.Compose([Flip(p=0.5),GridDistortion(p=0.5),GaussianBlur(3,0.15),ToTensorV2()])
    
    max_epochs = 75
    validation_freq = 1 # perform validation at each x epochs
    
    segmentationModel = initModel(device,useMultiGPUs=multiGPU,useVolContext=True)
    optimizer = initOptimizer(segmentationModel)

    projectDir = '/nfs/home/susana/deepLearning/challenges/LUNA16'    
    subsets = natsorted(glob.glob(os.path.join(projectDir,'subset*')))
    
    """
    At each fold, one subset will be used to test, another to perform validation and 8 for training
    We will ensure that the training and validation sets will be populated with the seriesUIDs of the respective
    subsets that evidence masked regions i.e. candidates (To this, the dictionary .pkl created by generateMasks.py will be useful)
    """
    # read the dictionary that encodes the UIDs and respective slices with candidates
    with open('/nfs/home/susana/deepLearning/challenges/LUNA16/maskedSlices.pkl', 'rb') as f:
        maskedSlices = pickle.load(f)

    # half-precision (speed-up)
    scaler = GradScaler()

    startingFold = 5
    for k_ in range(startingFold,10): # k-fold cross-validation loop
        outputData = dict()
        if k_ > startingFold: # refresh model between folds
            if not multiGPU:
                segmentationModel._init_weights()
            else:
                segmentationModel.module._init_weights()
        # Plot
        win = vis.line(X=np.array((0,)),Y=np.array(((1,))),env='main',win='lossWindow',name="train",opts=dict(ytickmin=0,ytickmax=1,xtickmin=0,xtickmax=max_epochs,title='Training & Validation Loss',xlabel='Epoch',ylabel='Soft Dice Loss'))
        vis.line(X=np.array((0,)), Y=np.array(((1,))),win=win,update="append",name="val",opts=dict(ytickmin=0,ytickmax=1,xtickmin=0,xtickmax=max_epochs,title='Training & Validation Loss',xlabel='Epoch',ylabel='Soft Dice Loss'))
        
        win2 = vis.image(np.zeros((512,512)).astype(np.uint8),env='main',win='SrcImg',opts=dict(caption='Random training slice',store_history=True))
        win3 = vis.image(np.zeros((512,512)).astype(np.uint8),env='main',win='SrcLabel',opts=dict(caption='Ground Truth Mask',store_history=True))
        win4 = vis.image(np.zeros((512,512)).astype(np.uint8),env='main',win='PLabel',opts=dict(caption='Predicted Mask',store_history=True))
        
        trainingLosses = []
        validationLosses = []

        print('--------------------Fold: %d--------------' % k_)
        trainingSet,validationSet,_ = createFolds(k_,subsets)
    
        trainingSamples = createListOfSamples(trainingSet,maskedSlices,filterByAnnotationsMask=True)
        validationSamples = createListOfSamples(validationSet,maskedSlices,filterByAnnotationsMask=True)

        print('#Masked Training Slices:\t %d' % len(trainingSamples))
        print('#Masked Validation Slices:\t %d' % len(validationSamples))

        # Generators
        trainingSet = Dataset(trainingSamples,transform=train_transform)
        training_generator = torch.utils.data.DataLoader(trainingSet, **trainingParams)

        validationSet = Dataset(validationSamples)
        validation_generator = torch.utils.data.DataLoader(validationSet, **trainingParams)
        
        # Loop over epochs
        for epoch_ in range(max_epochs):
            print('\n-------Epoch: %d---------' % epoch_)
            for batchIdx,batchVal in enumerate(training_generator):
                slices_batch,masks_batch = batchVal
                optimizer.zero_grad() 

                # Transfer to the GPU
                slices_batch = slices_batch.to(device)
                masks_batch = masks_batch.to(device)

                with autocast():
                    # Model computations (forward)
                    predictedSegmentation = segmentationModel(slices_batch) #predicted masks
                    
                    trainingLoss = diceLoss((predictedSegmentation,masks_batch))
                    trainingLoss = trainingLoss.mean() # compute batch-loss

                scaler.scale(trainingLoss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                gc.collect()

                if (batchIdx % 25) == 0:  # Display results at each 25 batches
                    sourceSlice = slices_batch[0,1,:,:].view((1,512,512)).cpu().data.numpy()
                    vis.image(revertNormalization(sourceSlice),win='SrcImg',opts=dict(store_history=True))
                    
                    # visualize ground truth
                    groundLabel = masks_batch[0,0,:,:].view((1,512,512)).cpu().data.numpy()
                    # visualize segmentation
                    vis.image(revertNormalization(groundLabel,numBits=1),win='SrcLabel',opts=dict(store_history=True))

                    segmentationVis = predictedSegmentation[0,0,:,:].view((1,512,512)).cpu().data.numpy()
                    # visualize segmentation
                    vis.image(revertNormalization(segmentationVis,numBits=1),win='PLabel',opts=dict(store_history=True))

                    del segmentationVis,sourceSlice,groundLabel
                
            print('\nTraining Loss:  %.4f' % trainingLoss)
            
            # Update plot
            vis.line(X=np.array((epoch_+1,)), Y=np.array(((trainingLoss.cpu().data.numpy(),))),win=win,update="append",name="train")
            trainingLosses.append(trainingLoss.cpu().data.numpy())
            
            if (epoch_ % validation_freq) == 0:
                # Perform Validation
                # accumulate batch losses
                validationBatchLosses = []
                print('----------------------Validation----------------------')
                with torch.set_grad_enabled(False): # Performs inference on validation data and prevents back-propagation
                    for slices_batch, masks_batch in validation_generator:
                        #debugPt = 1
                        #Transfer to GPU
                        slices_batch = slices_batch.to(device)
                        masks_batch = masks_batch.to(device)
                        
                        with autocast():
                            # Model computations
                            predictedSegmentation = segmentationModel(slices_batch) #predicted masks for the batch
                            # upsample to match the dimensions of the ground truth
                            #upsampler = torch.nn.Upsample(scale_factor = 2, mode='bicubic')
                            #predictedSegmentation = upsampler(predictedSegmentation)
                            validationloss = diceLoss((predictedSegmentation,masks_batch))
                            validationloss = validationloss.mean()
                            validationBatchLosses.append(validationloss.cpu().data.numpy())
                
                epoch_validationLoss = np.mean(validationBatchLosses)
                print('\nValidation Loss:  %.4f' % epoch_validationLoss)
                vis.line(X=np.array((epoch_+1,)), Y=np.array(((epoch_validationLoss,))),win=win,update="append",name="val")
                validationLosses.append(epoch_validationLoss)
                
                modelSuffix = 'fold'+str(k_)+'_epoch'+str(epoch_)+'.pth'
                torch.save(segmentationModel.state_dict(), '/nfs/home/susana/deepLearning/challenges/LUNA16/model2/checkpoints/'+modelSuffix)
                
        outputData['trainingLoss'] = trainingLosses
        outputData['validationLoss'] = validationLosses

        print('\nLearning curve created')
        outFile = open("/nfs/home/susana/deepLearning/challenges/LUNA16/model2/results/fold"+str(k_)+'.pkl', "wb")
        pickle.dump(outputData, outFile)
        outFile.close()
        print('\nLearning curve saved!')            