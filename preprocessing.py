""" ------------------------- Preprocessing script ------------------
Preprocessing aims to discard (zero-ing) all the entries of each slice that do not refer to lung tissue
To this, each slice is processed by:
    -> Standardization;
    -> K-Means clustering (K=2);
    -> Thresholding followed by binarization
    -> Morphological Operations (Erosion -> Dilation)
    -> Connected Component Analysis
The result is written to disk as a .npy file named {CT_UID}_slice_{sliceIndex}"""

import numpy as np
from joblib import Parallel,delayed
import multiprocessing as mp
import os,glob,cv2,pdb,sys,time
from natsort import natsorted
from sklearn.cluster import KMeans
from skimage import measure
import SimpleITK as sitk
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def createSliceFilename(dstDir,ctUID,idx):
    sliceID = ctUID+'_slice_'+str(idx)
    outputName = os.path.join(dstDir,sliceID)
    return outputName

def segmentSlice(args):
    """
    dims : tuple
        (height,width)
    dilationKernelSize : int
        Size of the kernel used to perform dilation and erosion. The default is 7.
    dstDir : str
        Top-level directory to store preprocessed data. The default is '/nfs/home/susana/deepLearning/challenges/LUNA16/data'.
    """
    sliceData,dims,dilationKernelSize = args
    numRows,numCols = dims
    
    debug=0
    if debug:
        plt.figure(1)
        plt.imshow(sliceData)
        
    img = np.array(sliceData - np.mean(sliceData)) # subtract the average value
    img = img / np.std(sliceData) # divide by the standard deviation

    # At this momente, we have standardized values
    
    intermediate_region = img[numRows//4:3*numRows//4,numCols//4:3*numCols//4] # define a window in the center of the image
    mean_ = np.mean(intermediate_region) # pick the average  
    max_ = np.max(intermediate_region) # pick the max
    min_ = np.min(intermediate_region) # pick the min 
    
    #Force extremely high and low intensities (air and bone) to the average 
    img[img==max_]=mean_
    img[img==min_]=mean_
        
    # # Kmeans with two clusters (high/low radiodensity)
    kmeans = KMeans(2).fit(np.reshape(intermediate_region.astype(np.float64),[np.size(intermediate_region),1]))
    clusters_centers = sorted(kmeans.cluster_centers_.flatten())
    
    # # Assuming a threshold "equally-distant" to each cluster center
    threshold = np.mean(clusters_centers)
    thresh_img = np.where(img<threshold,1,0).astype(np.uint8)  # binarization
    del img
    
    # Apply morphological opening
    thresh_img = cv2.erode(thresh_img,np.ones((3,3)),iterations=1)
    thresh_img = cv2.dilate(thresh_img,np.ones((5,5)),iterations=1)
    
    # Connected Component Analysis
    mask = measure.label(thresh_img) # Different labels are displayed in different colors
    # measure properties of each component
    labels_props = measure.regionprops(mask)
    
    for label_ in labels_props:
        B = label_.bbox # label bounding box
        hBbox = int(B[2]-B[0]) #bbox height
        wBbox = int(B[3]-B[1]) #bbox width
        if any([B[0] <= 0.10*numRows,B[2] >= 0.90*numRows,B[1] <= 0.10*numCols,B[3] >= 0.90*numCols]):
            # discard regions too close of the image limits
            mask[label_.coords[:,0],label_.coords[:,1]] = 0
        elif (hBbox < 50 or hBbox > 450) or (wBbox < 50 or wBbox > 450):
            # discard regions by size
            mask[label_.coords[:,0],label_.coords[:,1]] = 0
    
    mask[mask != 0] = 1
    mask = mask.astype(np.uint8)
    # Apply "defensive" dilation to preserve details close to the lung boundary
    mask = cv2.dilate(mask,np.ones((dilationKernelSize,dilationKernelSize)),iterations = 1) 
    
    # Apply standardization
    # The unmasked positions (value=0) will receive a new value, 
    # It will be calculated from the intensity distribution of the masked region (value=1)
    # The idea is to set air and "non-relevant" regions to low intensity values
    
    masked = np.array(sliceData)
    
    foregroundMean = np.mean(sliceData[mask > 0])
    foregroundStd = np.std(sliceData[mask > 0])
    if np.sum(mask > 0) > 0:
        replaceVal = foregroundMean - 1.75*foregroundStd
        masked[mask==0] = replaceVal # replace the background values

    # normalize
    finalMean = np.mean(masked)
    finalStd = np.std(masked)

    masked -= finalMean
    masked /= finalStd
    
    if debug:
        plt.figure(2)
        plt.imshow(masked)
    
    return masked

def segmentLungs(sequenceName,openingKernelSize,dstDir):
    """
    Parameters
    ----------
    sequenceName : string
        CT file
    Returns
    -------
    None. Writes to file the resulting slices of each CT
    """
   
    subset_ = sequenceName.split('/')[-2]
    ctUID_ = sequenceName.split('/')[-1][:-4]
    
    destination = os.path.join(dstDir,subset_)
    
    ct_data = sitk.ReadImage(sequenceName)
    ct_data = np.array(sitk.GetArrayFromImage(ct_data),dtype=np.float32)
    
    numSlices,numRows,numCols = ct_data.shape
    
    slices = [ct_data[x_,:,:] for x_ in range(len(ct_data))]

    openingKernelSize = [openingKernelSize]*len(slices)
    dimensions = [(numRows,numCols)]*len(slices)
    arguments = list(zip(slices,dimensions,openingKernelSize))
    numWorkers = int(0.9*mp.cpu_count())
   
    segmentedSlices = Parallel(n_jobs=numWorkers)(delayed(segmentSlice)(x) for x in arguments)
    del ct_data
    
    for sliceIdx,slice in enumerate(segmentedSlices):       
        outputFilename  = createSliceFilename(destination, ctUID_, sliceIdx)
        np.save(outputFilename,slice)
        
    return

def showPair(originalSlice,processedSlice):
    fig,axs = plt.subplots(1,2)
    axs[0].imshow(originalSlice)
    axs[1].imshow(processedSlice)
    return

def comparePair(ctName):
    
    orig_data = sitk.ReadImage(ctName)
    orig_data = np.array(sitk.GetArrayFromImage(orig_data),dtype=np.float32)
    
    # pick one random slice index
    sliceIdx = np.random.randint(0,orig_data.shape[0],1)[0]
    
    orig_data = orig_data[sliceIdx,:,:]

    # Find the processed slice
    subset_ = ctName.split('/')[-2]
    ctUID_ = ctName.split('/')[-1][:-4]

    sliceName = ctUID_+'_slice_'+str(sliceIdx)+'.npy'
    processedCT_name = os.path.join('/nfs/home/susana/deepLearning/challenges/LUNA16/data',subset_,sliceName)

    processed_data = np.load(processedCT_name)
    showPair(orig_data,processed_data)
    return

if __name__=='__main__':
    
    subsetIdx = int(sys.argv[1]) # process subset: 0,1, X
    
    DATA_GENERATION_MODE = 1
    CHECK_RESULTS_MODE = 0

    projectDir = '/nfs/home/susana/deepLearning/challenges/LUNA16'  # source data
    destinationDir='/nfs/home/susana/deepLearning/challenges/LUNA16/data'   # where to put output data

    subsets = natsorted(glob.glob(os.path.join(projectDir,'subset*')))
    subset_ = subsets[subsetIdx]

    # list all .mhd files
    cts_ = [os.path.join(subset_,x) for x in os.listdir(subset_) if x.endswith('mhd')]
    N = len(cts_)
    print('\nFound %d CT files' % N)

    if DATA_GENERATION_MODE:
        print('\nProcessing %s' % subset_.split('/')[-1])
        # Executation parameters
        openingKernelSize=9
       
        for i_,ct_ in enumerate(cts_):
            segmentLungs(ct_,openingKernelSize,destinationDir)    
            print('\nFinished (%d/%d)' %(i_+1,N))
        print('\nProcessed all CT files!')
    
    elif CHECK_RESULTS_MODE:
        # PICK ONE RANDOM CT file and sliceIdx to check visually
        ct_random_idx = np.random.randint(0,N,1)[0]
        targetCt = cts_[ct_random_idx]

        comparePair(targetCt)
        

        
        
            
