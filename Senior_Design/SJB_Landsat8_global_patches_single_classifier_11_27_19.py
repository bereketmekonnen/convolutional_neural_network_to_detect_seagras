# import python libray for cnn classification
import glob
import numpy as np
import spectral.io.envi as envi
import os
import scipy.io as sio
import pickle
from datetime import datetime
start = datetime.now()
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D
from keras.layers import Convolution2D, Convolution1D
from keras.models import load_model
import os
import numpy as np
import scipy.io as sio
from keras.utils import np_utils
import os.path
import h5py
import time
import matplotlib
from matplotlib import pyplot as plt
from tqdm.keras import TqdmCallback
from keras.layers.normalization import BatchNormalization
# from keras.utils import plot_model
import scipy as scipy
#from keras.utils import multi_gpu_model
import wandb
from wandb.keras import WandbCallback
# end of import section
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# this code is used to color the images.
from classification_label_to_image import classification_label2_image
#'10Oct13','26Oct13','14Nov14','30Nov14','16Oct15','02Oct16','06Nov17','25Sep19','28Nov19'

#hyperparamters for CNN
dimension = 3# patch size=3by3, range (1X1,3X3,5X5)
deleted_channels=[4,5,6] #example if you want delete channel 3 than select [2] or for 3,4= [2,3]# python index start from zero.
# variables
labels=[] # data variable holder 
x_train=[] # data variable holder 
selected_sample_per_class=15000 # number of patches for downsampling/upsampling  MAX/ MIN ( )
balanced_option='balanced' # select balanced/unbalanced 
senario='atm_cor'  #delete this line
numChannels =7 - len(deleted_channels) #number of sepctral channels
epochs = 200 # iterations
batchSize = 256 # 2 to power n 
numOfClasses = 5
#SENCITIVITY ANALYSIS
# wandb.init(project='SJB_cindy_global_model_12_6',name='Single_classifier_epoch_'+str(epochs)+'_dim_'+str(dimension),config={"hyper": "parameter"},reinit=True)
# not needed turn it on if you want it
#define CNN model here
def multi_gpu_cnn_model(numChannels):
    model = Sequential()
    if dimension == 1: #input patch size 1by 1
        print("Designing CNN for input 1x1x8")
        model.add(Convolution2D(32, (1, 1), activation='relu', input_shape=(dimension, dimension, numChannels)))
        model.add(Dropout(0.01))
        print(model.output)
        model.add(Convolution2D(16, (1, 1), activation='relu'))
        model.add(Dropout(0.01))
        print(model.output)
        model.add(Flatten()) # to make the data as a vector
        model.add(Dense(numOfClasses, activation='softmax')) # classification operation
        print(model.output)
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # adaptive learning rate optimization algorithm

    if dimension == 3: #input patch size 3by 3
        print("Designing CNN for input 3x3x8")
        model.add(Convolution2D(32, (1, 1), activation='relu', input_shape=(dimension, dimension, numChannels)))
        model.add(Dropout(0.01))
        print(model.output)
        model.add(Convolution2D(16, (3, 3), activation='relu')) 
        model.add(Dropout(0.01))
        print(model.output)
        model.add(Flatten()) # to make the data as a vector
        model.add(Dense(numOfClasses, activation='softmax'))# classification operation
        print(model.output)
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    if dimension == 5: #input patch size 5by 5
        print("Designing CNN for input 5x5x8")
        model.add(Convolution2D(32, (2, 2), activation='relu', input_shape=(dimension, dimension, numChannels)))
        model.add(Dropout(0.01))
        print(model.output)
        model.add(Convolution2D(16, (4, 4), activation='relu'))
        model.add(Dropout(0.01))
        print(model.output)
        model.add(Flatten())
        model.add(Dense(numOfClasses, activation='softmax'))
        print(model.output)
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# python libray to print output as text
import sys
orig_stdout = sys.stdout
# call the CNN model 
multi_model = multi_gpu_cnn_model(numChannels) #intialize CNN model parameters
# cropping start point of the defined ROIs
patch_crop_point=int(np.floor(dimension/2)) #for example if dimension is 3, patch_crop_point=1 #no padding applied
#define the directory name
save_directory = './global_model_3_2/'+'Single_classifier_del_chl_'+str(deleted_channels)+'_epochs_'+str(epochs)+'_dim_'+str(dimension)+'_spc_'+str(selected_sample_per_class)+'_batchS_'+str(batchSize)+'/'
#automatic directory creation
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    print('directory created', save_directory)
# declare CNN filename
cnnFileName = 'Single_classifier_'+balanced_option+'_dimension_' + str(
    dimension) + "_channels_" + str(
    numChannels) + "_epochs_" + str(epochs) + "_batchSize_" + str(batchSize) + "numOfClasses" + str(
    numOfClasses) + ".h5"
model = Sequential()
# if model is already trained it will skip the training
if os.path.isfile(save_directory+cnnFileName):
	# sequential keras model declare
    model = Sequential()
    multi_model = load_model(save_directory+cnnFileName)
    multi_model.summary()
else:
	# different class name, can be increased
    class_name = ["Deep_Water", "Sand", "Seagrass", "Land", "Intertidal"] # fiveclasses for SJB
    print(class_name)
	# this for loop will go trough each class seperately to extract the specified patch size across all temporal images
    for class_numb in range(0, len(class_name)):
		# 10 temporal images taken from Saint Joseph Bay Landsate Image.
        locations = ['10Oct13']
        for iter_locat in range(0, len(locations)):
            location = locations[iter_locat]
			# Training ROIs data directory
            training_data_directory="./Landsat8_for_Classification/"+location+"/ROIs for Training Global/"
			# this will identify number of ROIs exists per class and per temporal image
            A = (glob.glob(training_data_directory + class_name[class_numb] + "_*.hdr"))
            print(A)
            sample_data = [] # variable holder declared
            for patchnumb in range(0, len(A)):
				# python spectral libray will read the envi image
                lib = envi.open(A[patchnumb], A[patchnumb][0:-4])
                print(lib)
                im = lib
				# Following line of code to discard the boundary pixel of the Image
                a_zeros = np.zeros([lib.shape[0], lib.shape[1]])
                if dimension>1:
                    a_zeros[0:, 0:patch_crop_point] = 1
                    a_zeros[0:patch_crop_point, 0:] = 1
                    a_zeros[-patch_crop_point:, 0:] = 1
                    a_zeros[0:, -patch_crop_point:] = 1

                indeex_loc = np.where(a_zeros == 0)
				# End of discard the boundary pixel of the Image
                rows_loc = indeex_loc[0] # number of rows
                colms_loc = indeex_loc[1] #number of columns
                print('rows_loc colms_loc', lib.shape[0], lib.shape[1], colms_loc.shape[0], rows_loc.shape[0])
                data_divided_into = 1 # all the training rois divided into 1 but it can be divided into more parts to improve memory constraint
                length_all_location = int(rows_loc.shape[0])
                division_len = int(np.ceil(length_all_location / data_divided_into));
                count_data_division = 0
                for iteration in range(0, length_all_location, division_len):
                    print('code running')
                    count_data_division = count_data_division + 1
                    print('division number', count_data_division, '    :.....')
                    print('division number', length_all_location, '    :.....', )
                    if count_data_division == data_divided_into:
                        data_iter_end = length_all_location;
                    else:
                        data_iter_end = iteration + division_len;

                    data_length = (data_iter_end - iteration);
					# declare the patch size and number of patches
                    f = np.zeros([data_length, dimension, dimension, lib.shape[2]]);
                    image_index = np.zeros([data_length, 2]);

                    for data_iter in range(iteration, data_iter_end):
                        l = rows_loc[data_iter]; # extract the patch crop point row
                        m = colms_loc[data_iter]; # extract the patch crop point column
                        e = np.zeros([dimension, dimension,  lib.shape[2]]); # patch size , patch size, number of bands
						# extracting patches 
                        e[0:dimension, 0:dimension,0: lib.shape[2]] = im[l -patch_crop_point:l+patch_crop_point+1, m -patch_crop_point:m+patch_crop_point+1, :];
                        image_index[(data_iter - iteration), :] = [l, m]; # saving co ordinate information for patch location
                        f[(data_iter - iteration), :, :, :] = e; # combining all the patches
                sample_data.append(f) #concatenating all the patches from the same ROIs
            sample_data = np.concatenate(sample_data) # concatenating all the patches from all ROIs from same class

            print('Sample data Shape', sample_data.shape)
            if balanced_option == 'balanced':
				# randomly upsampled or downsampled to balance the data
                if sample_data.shape[0] > selected_sample_per_class:
                    index_downsample = np.random.choice(sample_data.shape[0], selected_sample_per_class, replace=False)
                elif sample_data.shape[0] < selected_sample_per_class:
                    index_downsample = np.random.choice(sample_data.shape[0], selected_sample_per_class)
                sample_data = sample_data[index_downsample, :, :, :]
                print('after blancing Sample data Shape', sample_data.shape)
                labels.append(class_numb * np.ones(selected_sample_per_class))
                x_train.append(sample_data)
                del sample_data
            elif balanced_option == 'unbalanced':
				# unbalanced dataset
                labels.append(class_numb * np.ones(int(sample_data.shape[0])))
                x_train.append(sample_data)
                del sample_data
	# concatenate all the training dataset
    x_train = np.concatenate(x_train)
	# same number of labels
    labels = np.concatenate(labels);
    x_train = np.delete(x_train, deleted_channels, 3)
    print(x_train.shape, labels.shape)
    y_train = np_utils.to_categorical(labels) # one hot vector conversion
    f = open(save_directory +'command_window_'+'.txt', 'w')
    # sys.stdout = f
	# split the data intro train and test dataset, shuffle the training data before each epochs,
    print('starting the traninig of the model')
    history=multi_model.fit(x_train, y_train, epochs=epochs, batch_size=batchSize,validation_split=0.1, shuffle=True, verbose=0, callbacks=[TqdmCallback(verbose=0)])
    print(history)
    pickle.dump(history, open(save_directory + location + '_results_' + location + '.p', "wb"))
    # sys.stdout = orig_stdout
    print(history.history.keys())
    print(history)
    #  "Accuracy" curve save in png format for training and validation
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_directory+'acc.png')
    plt.clf()
    # "Loss" curve save in png format for training and validation
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_directory+'loss.png')
    multi_model.save(save_directory+cnnFileName) # save the CNN classifier in the directory
    plt.clf()
    # multi_model.summary()
#training Ends
#testing begins
# 10 temporal images will be classified using a loop
locations = ['10Oct13']
for iter_locat in range(0, len(locations)):
    location = locations[iter_locat]
    method_name = location + '.mat'
				# Testing image data directory
    training_data_directory = "./Landsat8_for_Classification/" + location+"/"
	# read the testing image header file and filename
    A = (glob.glob(training_data_directory +  "*.hdr"))
	#load the testing image
    lib = envi.open(A[0], A[0][0:-4])
    im=lib
	#removing the boundary pixels
    a_zeros=np.zeros([lib.shape[0],lib.shape[1]])
    if dimension>1:
        a_zeros[0:, 0:patch_crop_point] = 1
        a_zeros[0:patch_crop_point, 0:] = 1
        a_zeros[-patch_crop_point:, 0:] = 1
        a_zeros[0:, -patch_crop_point:] = 1
    c=np.where(lib[:,:,[0,1,2]]==0)
    rows_half = lib.shape[0];  # columns=11134; rows=4403;
    colms_half = lib.shape[1];
		#removing the boundary pixels ends
			#removing the zero value pixels
    nan_values_location=np.where(lib[:,:,[0,1,2,3,4,5]]>0)
    testing_result_class1 = np.zeros([rows_half, colms_half]); # testing variable declare
    print('nan_values_location',nan_values_location[0].shape)
    print(testing_result_class1.shape)
    # 3591, 3591
    indeex_loc = np.where(a_zeros == 0) # finding the co ordinate of patch extraction
    rows_loc=indeex_loc[0] # number of row 
    colms_loc=indeex_loc[1] # number of column
    print(lib,rows_loc.shape,colms_loc.shape,a_zeros.shape)
    data_divided_into=100 #dividing the testing image into 100 parts to solve memory constraint
    length_all_location=int(rows_loc.shape[0])
    division_len = int(np.ceil(length_all_location / data_divided_into));
    print('division_len',division_len)
    count_data_division = 0
    for iteration in range(0, length_all_location,division_len):
        print('code running')
        count_data_division = count_data_division + 1
        print('division number',count_data_division,'    :.....')
        if count_data_division == data_divided_into:
            data_iter_end = length_all_location;
        else:
            data_iter_end = iteration + division_len;

        data_length = (data_iter_end - iteration) ;
		# declaring the testing variable with number of patches,  patch size and channel information
        f = np.zeros([data_length,dimension, dimension,  lib.shape[2]]);
        image_index = np.zeros([ data_length,2]);
        for data_iter in range (iteration,data_iter_end):
            l = rows_loc[data_iter];
            m = colms_loc[data_iter];
            e = np.zeros([dimension, dimension, lib.shape[2]]);
			# extracting the image patch
            e[0:dimension, 0:dimension,0: lib.shape[2]] = im[l -patch_crop_point:l + patch_crop_point+1, m - patch_crop_point:m + patch_crop_point+1,0 : lib.shape[2]];
            image_index[ (data_iter - iteration),:]=[l, m]; # saving the image row and column infomatipn
            f[(data_iter - iteration),:,:,: ]=e; # saving the ROIs into a variable
		#predicting the class label for testing images
        f = np.delete(f, deleted_channels, 3)
        predicted_label = multi_model.predict(f, batch_size=1024)
        print(predicted_label.shape)
        y_pred_arg = np.argmax(predicted_label, axis=1); # labeling image with class label which has highest probability
        print(y_pred_arg.shape)
		# converting back to image shape
        test_location = image_index 
        predict_label_len = len(y_pred_arg)
        for index_predict in range(0, predict_label_len):
            testing_result_class1[int(test_location[ index_predict,0]), int(test_location[ index_predict,1])] =  y_pred_arg[index_predict]+1 ;
	#removing the nan values 
    testing_result_class2 = np.zeros([rows_half, colms_half]);
    for i in range(len(nan_values_location[0])):
        testing_result_class2[nan_values_location[0][i], nan_values_location[1][i]] = 1
    testing_result_class1=np.multiply(testing_result_class1,testing_result_class2)
    testing_result_class1 = np.uint8(testing_result_class1)
		#removing the nan values  end
    sio.savemat(save_directory + location + '_results_' + method_name, {'testing_result_class1': testing_result_class1}) # save image as a matlab file 
    difference = datetime.now() - start
    print(difference) # calculate the time
    im3 = classification_label2_image(testing_result_class1,save_directory + location + '_' + method_name[0:-4]) # save as a image file
    difference = datetime.now() - start
    print(difference)
    difference = datetime.now() - start
    print(difference); # calculate the time





