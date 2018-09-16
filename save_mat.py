
"""

Note::: In order for this code to work, you need to have face and nonface images created using the "data_creation.py" file
Description::: This code was used to extract the 60x60 cropped images, flatten into vectors (1 x 10800)
 and save them as mat file, along with their labels
"""


import numpy as np
#import csv
import cv2
#import os
import glob
import scipy.io
from sklearn import preprocessing
##
def data_collection():
    path_nface=  'Path/nonface_images'
    path_face = 'Path/face_images'
    t = np.zeros([11000,10800])
    u = np.zeros([11000,10800])
    b=0
    for img in glob.glob(path_face+'/*.jpg'):
        if b<11000:
            face= cv2.imread(img)
            t[b,:] = np.reshape(face, (1,10800))
        else:
            break
        b +=1
    a=0
    for img in glob.glob(path_nface+'/*.jpg'):
        if a <11000:
            nonface=cv2.imread(img)
            u[a,:] = np.reshape(nonface, (1,10800))
        else:
            break
        a+=1        
    x=np.concatenate((t,u),axis=0)
    y1=np.concatenate((np.zeros([11000,1]),np.ones([11000,1])),axis=0)
    y2=np.concatenate((np.ones([11000,1]),np.zeros([11000,1])),axis=0)
    y=np.concatenate((y1,y2),axis=1)
    return x,y

total_data, total_label=data_collection()
total_data_copy= total_data
#PreProcess Data
scalar = preprocessing.StandardScaler()  
scalar.fit(total_data)
preprocessing.StandardScaler(copy=True, with_mean = True, with_std= True)  
total_data=scalar.transform(total_data)
#Save into Mat Files
a= scipy.io.savemat('total_label.mat',{'total_label':total_label})
b= scipy.io.savemat('total_data.mat',{'total_data':total_data}) #Preprocesssed, Zero Centered Data
c= scipy.io.savemat('total_data_copy.mat',{'total_data_copy':total_data_copy}) # Non Pre Processed original data
