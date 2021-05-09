import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import spectral.io.envi as envi
location ='10Oct13'
training_data_directory = "./Landsat8_for_Classification/" + location+"/"
training_data_directory_1 = "./Landsat8_for_Classification/"+location+"/ROIs for Training Global/"
A = (glob.glob(training_data_directory +"*.hdr"))
# A = (glob.glob(training_data_directory_1 +"Seagrass"  +"_*.hdr"))
#load the testing image
lib = envi.open(A[0], A[0][0:-4])
lib.shape
im=lib

temp = np.zeros([im.shape[0],im.shape[1],3])
temp1 = im.asarray()
temp[:,:,2]=temp1[:,:,3]
temp[:,:,1]=temp1[:,:,2]
temp[:,:,0]=temp1[:,:,1]
mx,mn = temp.max(), temp.min()
oldr = mx-mn
n = (255/oldr)
temp = ( (temp - mn)* n)
temp = temp.astype(np.uint8)
cv2.imwrite("orginal_image.jpg",temp)
temp = temp*1000
temp = temp.astype(np.uint8)

temp2 = temp
temp_hsv = cv2.cvtColor(temp2, cv2.COLOR_RGB2HSV)
temp_hsv[:,:,2] = np.where(temp_hsv[:,:,2]>100,100,temp_hsv[:,:,2])
temp2 = cv2.cvtColor(temp_hsv, cv2.COLOR_HSV2RGB)
# temp2 = np.where(temp2<50,0,temp2)
mx,mn = temp2.max(),temp2.min()
oldr = mx-mn
n = (255/oldr)
temp2 = ( (temp2 - mn)* n)+0
temp2 = temp2.astype(np.uint8)
temp2.max()
temp2.min()

temp_yuv = cv2.cvtColor(temp, cv2.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
temp_yuv[:,:,0] = cv2.equalizeHist(temp_yuv[:,:,0])
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(temp_yuv, cv2.COLOR_YUV2BGR)

np.where()
temp_hsv = cv2.cvtColor(temp2, cv2.COLOR_HSV2RGB)

temp_hsv = cv2.cvtColor(temp2, cv2.COLOR_RGB2HSV)
# Histogram equalisation on the V-channel
temp_hsv[:, :, 2] = cv2.equalizeHist(temp_hsv[:, :, 2])
# convert image back from HSV to RGB
image_hsv = cv2.cvtColor(temp_hsv, cv2.COLOR_HSV2RGB)
cv2.imshow("temp",temp)
cv2.imshow("temp2",temp2)
cv2.waitKey()
cv2.imwrite('testing_image.jpg',temp2)
cv2.imshow("equ",img_output)
cv2.imshow("equ_hsv",image_hsv)

def overlayonorginal(clsfied):
    # clsfied = cv2.imread("./global_model_3_2/Single_classifier_del_chl_[]_epochs_500_dim_3_spc_15000_batchS_256/10Oct13_10Oct13_cv2.png")
    orgimg = cv2.imread("./Landsat8_for_Classification/10Oct13/testing_image.jpg")
    # orgimg = temp
    a=0.93
    b=1-a
    if(orgimg.shape != clsfied.shape):
        orgimg = cv2.resize(orgimg,clsfied.shape[1::-1])
    dst = cv2.addWeighted(orgimg, a, clsfied, b, 0.0)
    # cv2.imwrite("dim3_500E,SS15k_overlayimage.jpg",dst)
    cv2.imshow("overlay", dst)
    cv2.waitKey()
    return dst
from PIL import Image, ImageEnhance
m =Image.open("orginal_image.jpg")
br = ImageEnhance.Brightness(m)
m1 = br.enhance(1.9)
m1 = ImageEnhance.Contrast(m1).enhance(1.2)
m1.save('testing_image.png.png')
m1.show('bright')
cv2.imshow('bright',np.array(m1.convert('RGB')))
cv2.waitKey()
cv2.imwrite("testing_image.png",np.array(m1))

m1 = Image.open('testing_image.png')
m2 = Image.open('over.png')
m3 = Image.blend(m1,m2,0.15)
# cv2.imshow('bright',np.array(m1))
m3.save("mmmmm.png")
plt.imshow(np.array(m3))
cv2.waitKey()

t= cv2.imread('testing_image.png')
cv2.imshow("  ",t)
cv2.waitKey()
