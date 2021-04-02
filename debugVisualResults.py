#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 23:52:46 2021

@author: susana
"""
### Show segmentation results
import numpy as np
from matplotlib import pyplot as plt
import os
from preprocessing import segmentSlice
import SimpleITK as sitk

srcFolder = '/nfs/home/susana/deepLearning/challenges/LUNA16/subset0'
filesInFolder = [x for x in os.listdir(srcFolder) if x.endswith('mhd')]
numFilesInFolder= len(filesInFolder)

randomIdx = np.random.randint(0,numFilesInFolder,1)[0]
sequenceName = os.path.join('/nfs/home/susana/deepLearning/challenges/LUNA16/subset0',filesInFolder[randomIdx])
ct_data = sitk.ReadImage(sequenceName)
ct_data = np.array(sitk.GetArrayFromImage(ct_data),dtype=np.float32)

randomSlice = np.random.randint(0,len(ct_data),1)

ksize=9
dims=(512,512)

slice_data = np.array(ct_data[randomSlice,:,:])
slice_data = np.squeeze(slice_data)
slice_preprocessed = segmentSlice((slice_data,dims,ksize))

