""" ------------------------- Generate Masks script ------------------
The code runs through the annotations.csv file:

    -> group instances/nodules by seriesUID    
    -> each seriesUID is processed independently:
        -> initialization of an empty volume;
        -> filling ones in nodule positions;
        -> write each binary slice as a file 
    
The result is written to disk as a .npy file named {CT_UID}_mask_{sliceIndex}
"""

import sys,csv,glob,os
from joblib import Parallel,delayed
import numpy as np
import SimpleITK as sitk
import multiprocessing as mp
import pickle

def createSliceMaskFilename(dstDir,ctUID,idx):
    sliceID = ctUID+'_slice_'+str(idx)+'_mask'
    outputName = os.path.join(dstDir,sliceID)
    return outputName

def createMask(dstArray,centerInVoxels,diameter,spacing):
    
    radius_x =  np.rint((diameter/spacing[0]+1)/2)
    radius_y = np.rint((diameter/spacing[1]+1)/2)
    radius_z = np.rint((diameter/spacing[2]+1)/2)
    
    x_min = int(centerInVoxels[0] - radius_x)
    x_max = int(centerInVoxels[0] + radius_x + 1)
    
    y_min = int(centerInVoxels[1] - radius_y)
    y_max = int(centerInVoxels[1] + radius_y + 1)
    
    z_min = int(centerInVoxels[2] - radius_z)
    z_max = int(centerInVoxels[2] + radius_z + 1)
    
    #print('X: [%d;%d]\nY: [%d;%d]\nZ: [%d;%d]' %(x_min,x_max,y_min,y_max,z_min,z_max))
    
    dstArray[z_min:z_max,y_min:y_max,x_min:x_max] = 1
    
    # return a tuple indicating the slices with positive responses (mask == 1)
    return np.arange(z_min,z_max)

def generate2DMasks(args):

    dstDir = '/nfs/home/susana/deepLearning/challenges/LUNA16/data'
    
    seriesUID = args[0][0]
    # find the absolute path to the sequence file
    sequenceName = glob.glob('/nfs/home/susana/deepLearning/challenges/LUNA16/subset*/{}.mhd'.format(seriesUID))[0]
    
    subset_ = sequenceName.split('/')[-2]
    dstDir = os.path.join(dstDir,subset_,'masks')
    
    # read the sequence metadata
    sequence = sitk.ReadImage(sequenceName)    
    numSlices = sequence.GetDepth()
    numRows = sequence.GetHeight()
    numCols = sequence.GetWidth()
    spacing = np.array(sequence.GetSpacing()) #xyz
    origin = np.array(sequence.GetOrigin()) #xyz
    
    # initialize an empty volume
    mask = np.zeros((numSlices,numRows,numCols)).astype(np.bool)
    
    slicesOfInterest = []

    for candidate_ in args: # cycle all the candidates belonging to this seriesUID
        _, c_x, c_y,c_z, diameter = candidate_ 
        center = np.array([float(c_x),float(c_y),float(c_z)]) # xyz (mm)
        center = np.rint((center - origin) / spacing) #xyz (voxels)
        
        z_interest = createMask(mask, center, float(diameter),spacing)
        slicesOfInterest.extend(z_interest)

    # write all slices
    for sliceIdx in range(numSlices):
        outputFilename  = createSliceMaskFilename(dstDir, seriesUID, sliceIdx)
        np.save(outputFilename,mask[sliceIdx,:,:])
    
    return (sequenceName,numSlices,list(set(slicesOfInterest)))
        
if __name__=='__main__':

    annotationsFile = sys.argv[1]
    numWorkers = int(0.9*mp.cpu_count())

    with open(annotationsFile) as f:
        annotations = [tuple(line_) for line_ in csv.reader(f)]
    
    del annotations[0]

    # At this point, each df_nodule is a tuple (seriesUid,centerX,centerY,centerZ,diameter)
    N_candidates = len(annotations)
    print('Found %d nodule candidates' % N_candidates)
    groupedCandidates = [] # Create a list of lists; each sublist refer to a given seriesuid
    
    # Find the unique set of seriesuid's
    seriesUids = list(set([x[0] for x in annotations]))

    for serieuid_ in seriesUids:
        sameUID_instances = [(idx_,val_) for idx_,val_ in enumerate(annotations) if val_[0] == serieuid_]
        sameUID_instances_idxs,sameUID_instances_nodules = list(zip(*sameUID_instances))
        
        groupedCandidates.append(sameUID_instances_nodules)
        
        # sort idxs by decreasing value and remove such entries => speeds-up the process for future iterations 
        sameUID_instances_idxs = sorted(sameUID_instances_idxs,reverse=True)

        for idx_ in sameUID_instances_idxs:
            del annotations[idx_]
        
        generate2DMasks(groupedCandidates[0])
    
    # each element of groupedCandidates refer to a single seriesuid and can include more than one candidate
    # the creation of the masks will be performed in parallel (each worker will deal with a seriesuid)

    # After this execution, each slice of the seriesUID present in the annotations.csv will be written to file
    segmentedSlices = Parallel(n_jobs=numWorkers)(delayed(generate2DMasks)(x) for x in groupedCandidates)
    
    #ALSO, segmented slices is a list of tuples, each including
        # 1st position - ctName;
        # 2nd position - # of slices that the ct file includes;
        # 3nd position - slices of interest (with masked positions)

    processedCTs = [x[0] for x in segmentedSlices]
    numSlices = [x[1] for x in segmentedSlices]
    maskedSlices = [x[2] for x in segmentedSlices]
    
    slicesOfInterest_ = dict.fromkeys(processedCTs, [])
    slicesOfInterest_ = { val_: (numSlices[idx_],maskedSlices[idx_]) for idx_,val_ in enumerate(slicesOfInterest_)}
    
    # Write this dictionary to disk (it will help sampling training/validation examples)
    
    f = open('/nfs/home/susana/deepLearning/challenges/LUNA16/maskedSlices.pkl','wb')
    pickle.dump(slicesOfInterest_,f)
    f.close()