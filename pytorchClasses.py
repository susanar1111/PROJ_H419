import torch,math
from torch import nn as nn
from unet import UNet
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt

class Dataset(torch.utils.data.Dataset):
    
    'Characterizes a dataset for PyTorch'
    def __init__(self, samples,useVolCtxt=True,transform=None):
        imgs, masks = list(zip(*samples))
        assert len(imgs) == len(masks)
        self.m_slices = imgs
        self.m_masks = masks
        self.m_useVolContext = useVolCtxt
        self.transform = transform
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.m_slices)

    def __getitem__(self, index):
        'Generates one sample of data'
        slice = np.load(self.m_slices[index]) #this is the slice that we want to segment
        mask = np.load(self.m_masks[index]) # its corresponding mask
        
        subset_ = self.m_slices[index].split('/')[-2]
        ct_uid = (self.m_slices[index].split('/')[-1]).split('_')[0]
        
        if self.m_useVolContext:
            #we need to load also the n-1 and n+1 slices as the neural network receives a "pseudo-colored" image given by three grayscale bands
            sliceIdx = self.m_slices[index].split('_slice_')[1]
            sliceIdx = int(sliceIdx[:-4])

            previousSliceIdx = max(0,sliceIdx-1)
            previous_slice_file = os.path.join('/nfs/home/susana/deepLearning/challenges/LUNA16/data',subset_,ct_uid+'_slice_'+str(previousSliceIdx)+'.npy')
            previous_slice = np.load(previous_slice_file)
            
            nextSliceIdx = sliceIdx + 1
            next_slice_file = os.path.join('/nfs/home/susana/deepLearning/challenges/LUNA16/data',subset_,ct_uid+'_slice_'+str(nextSliceIdx)+'.npy')
            
            if not os.path.exists(next_slice_file):
                next_slice = np.array(slice) #if slice refers to the last slice of the sequence just repeat it as the next one
            else:
                next_slice = np.load(next_slice_file)

            slice = np.dstack((previous_slice,slice,next_slice))

        if self.transform is not None: # perform augmentation
            # Images were saved as float32 after preprocessing;
            # For np.float32 input, Albumentations expects that value will lie in the range between 0.0 and 1.0.

            # Apply offset to get all values in the range ]0;...
            offset_ = np.amin(slice)
            slice += offset_
            
            transformed = self.transform(image=(slice / np.amax(slice)), mask=mask.astype(np.uint8))
            
            slice = transformed["image"] # those will be tensors
            mask = transformed["mask"]
            slice = slice.float()
            mask = mask.unsqueeze(0)
            mask = mask.float()  #if casted to bool the gradients will not be computed and backpropagated
        else:
            # change the numpy arrays to something pytorch recognizes properly (albumentations does this for us if we use augmentation)
            # Namely, torch image assumes axes in the form channel x numRows x numCols
            if self.m_useVolContext:
                slice = np.transpose(slice,(2,0,1)) # (3slices,#Rows,#cols)
            else:
                slice = np.expand_dims(slice,axis=0) #add one dim (1,#Rows,#cols)
            
            mask = np.expand_dims(mask,axis=0) #add "channel" dimension to the mask tensor to match the shape of slice

            slice = torch.from_numpy(slice)        
            mask = torch.from_numpy(mask).float() # if casted to bool the gradients will not be computed and backpropagated
        
        return slice,mask
    

class CustomUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels']) # applies normalization to the batch
        self.unet = UNet(**kwargs) # reuse the implementation of U-NET
        self.finalLayer = nn.Sigmoid()
        #self.finalLayer = nn.LeakyReLU() # this will allow to map feature values to the range [0;1] in a single channel
        #self._init_weights()
    
    def _init_weights(self): # Adopted from "Deep Learning with Pytorch"
        # This function initializes the filters used in the convolutional layers of U-NET
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu', a=0)
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
    
    def forward(self,inputBatch):
        batchOutput = self.input_batchnorm(inputBatch)
        unetOutput = self.unet(batchOutput)
        finalOutput = self.finalLayer(unetOutput)
        return finalOutput
