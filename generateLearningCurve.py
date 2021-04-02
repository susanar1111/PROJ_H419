from matplotlib import pyplot as plt
import sys,pickle
import numpy as np

debug = False
if not debug:
    resultsFile = sys.argv[1]
else:
    resultsFile = '/nfs/home/susana/deepLearning/challenges/LUNA16/model3/results/fold1.pkl'

trainValidationFile = open(resultsFile, 'rb')
trainValidationResults = pickle.load(trainValidationFile)
trainValidationFile.close()

N_epochs = len(trainValidationResults['trainingLoss'])

plt.plot(np.arange(75)+1,trainValidationResults['trainingLoss'][:75],marker='',color='blue',linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Soft Dice Loss')
plt.plot(np.arange(75)+1,trainValidationResults['validationLoss'][:75],marker='',color='orange',linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Soft Dice Loss')




