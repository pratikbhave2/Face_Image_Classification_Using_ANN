"""
Note : This is not a self contained code and it needs the path of the entire CelebA dataset,
along with the path of the annotation file.
Based on the annotation of the bounding box, face images are cropped and resized to 60x60
Nonface images are created slicing the left top corner of the image of size 60x60
Images used for training = 10000 for face and non-face each
Images used for testing = 1000 for face and non-face each
Total Images = 22000

"""


import os
import cv2
from pathlib import Path


# set path to the folder which contains the data annotation text file 
path=Path('Path_to_CelebA_images_folder')
os.chdir(path)
with open('list_bbox_celeba.txt','r') as txt:
    c=0
    for row in txt:
        if c<11000:    
            l= row.split()
	    #Set path to folder which contains raw images
            os.chdir(path/"images"/"img_celeba"/"img_celeba")
            img=cv2.imread(l[0])
            x_1=int(l[1])
            y_1=int(l[2])
            w=int(l[3])
            h=int(l[4])
            crop_img=img[y_1:y_1+h,x_1:x_1+w]
            resized_image = cv2.resize(crop_img, (60, 60),interpolation = cv2.INTER_AREA)
            
#	    #Set path to the folder where you want to write the cropped face images
            os.chdir(path/"face_images")
            cv2.imwrite(l[0], resized_image)
            os.chdir(path/"nonface_images")
            newimg=img[0:60,0:60,:]
            cv2.imwrite(l[0],newimg)
        
        c+=1
    
        
