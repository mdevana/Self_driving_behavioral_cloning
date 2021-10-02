#training the model
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
import math
import random

# imports for Keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

def ReadTrainingData(pathtocsv):
    #read lines of data from given driving_log.csv File 
    # Store it in list and remove the fist element
    
    lines = []
    with open (pathtocsv) as csvfile:
	    reader = csv.reader(csvfile)
	    for line in reader:
		    lines.append(line)
    lines.pop(0)
    return lines

def getRGBImage(pathtoimage):
    # Extract filename and folder name of image. The excel file has contents in Windows format
    # Form the new path according to linux format
    # Read the image from file and return in RGB format 
    
    filename = pathtoimage.split('\\')[-1]
    #print(filename)
    foldername = pathtoimage.split('\\')[-3]
    #print(foldername)
    
    new_path = MainPath + '/' + foldername + '/IMG/' + filename
    img_to_load = cv2.imread(new_path)
    img_to_load = cv2.cvtColor(img_to_load, cv2.COLOR_BGR2RGB)
    
    return img_to_load

def im_crop(img_full,x_crop,y_crop):
    # Crop and return image according to crop value given height and width.
    # The given value in height and width axis is applied on top, botton and sides respectively
    img_height=img_full.shape[0]
    img_width=img_full.shape[1]
    
    y_start = y_crop
    y_end = img_height - y_crop
    
    x_start = x_crop
    x_end = img_width - x_crop
    
    img_cropped = img_full[y_start:y_end, x_start:x_end]    
    
    return img_cropped

def im_aug_bright(img):
    # Function to randomly alter brightness of image using HSV Format
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    img_hsv[:,:,2] = img_hsv[:,:,2] * (np.random.uniform(0.25,1.0))
    
    #Leave it in HSV Format
    #img_bright = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    return img_hsv
    

def im_preprocess(img_to_process,angle):
    # Pipeline function to include the preprocessing steps required for the data    
    steer_angle = angle
    img = img_to_process         
    
#    cv2.imwrite("test_raw_image.jpg",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # Crop the images to remove unwanted areas. No crop at preprocessing stage. 
    # Cropping is integrated in the model.
    img = im_crop(img,0,0)
    
#    cv2.imwrite("test_cropped_image.jpg",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Flip images randomly to reduce bias to left or right turn. flip steer angle as well. 
    if (np.random.uniform() > 0.5):      
        img = cv2.flip(img,1)
        steer_angle = -1 * steer_angle
        
#    cv2.imwrite("test_flip.jpg",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # Randomly change brightness. Not used in this pipeline
    #img = im_aug_bright(img)
    
    # Resize the image is not used as no cropping is done
    #img = cv2.resize(img, (200,66))
    
#    cv2.imwrite("test_cropped_bright.jpg",cv2.cvtColor(img, cv2.COLOR_HSV2BGR))

    
    return img,steer_angle

def Choose_image_from_camera(line):
    # Augment data set by including data from left and right camera.
    # Return a list of 3 images.
    
    
    steer_center = float(line[3])
    Factor_Steering_Correction = 0.21
    imgs=[]
    angles=[]
    
    img_to_load = getRGBImage(line[0].strip())
    steer_angle = steer_center
    img_prep,s_angle = im_preprocess(img_to_load,steer_angle)
    imgs.append(img_prep)
    angles.append(s_angle)
    
    img_to_load = getRGBImage(line[1].strip())
    steer_angle = steer_center +  Factor_Steering_Correction
    img_prep,s_angle = im_preprocess(img_to_load,steer_angle)
    imgs.append(img_prep)
    angles.append(s_angle)
    
    img_to_load = getRGBImage(line[2].strip())
    steer_angle = steer_center -  Factor_Steering_Correction
    img_prep,s_angle = im_preprocess(img_to_load,steer_angle)
    imgs.append(img_prep)
    angles.append(s_angle)

    return imgs, angles
    

# Configure the paths to tranining data
MainPath_home = '/home/workspace/CarND-Behavioral-Cloning-P3'
MainPath = MainPath_home

DataPath_1 = MainPath + '/Forward_Drive/'+ 'driving_log.csv'
DataPath_2 = MainPath + '/Backward_Drive/'+ 'driving_log.csv'
DataPath_3 = MainPath + '/Forward_Recovery/'+ 'driving_log.csv'
DataPath_4 = MainPath + '/Backward_Recovery/'+ 'driving_log.csv'
DataPath_5 = MainPath + '/Track2_Forward/'+ 'driving_log.csv'
DataPath_6 = MainPath + '/Forward_Curves/'+ 'driving_log.csv'
DataPath_7 = MainPath + '/Backward_Curves/'+ 'driving_log.csv'

DataSet_1 = ReadTrainingData(DataPath_1)
DataSet_2 = ReadTrainingData(DataPath_2)
DataSet_3 = ReadTrainingData(DataPath_3)
DataSet_4 = ReadTrainingData(DataPath_4)
DataSet_5 = ReadTrainingData(DataPath_5)
DataSet_6 = ReadTrainingData(DataPath_6)
DataSet_7 = ReadTrainingData(DataPath_7)

# Merge the training data from different folders.
lines = DataSet_1 + DataSet_2 + DataSet_3 + DataSet_4 + DataSet_5 + DataSet_6 + DataSet_7 

print('Total numbers of lines read :'+ str(len(lines)))

# testing for preprocessing code 
#for i, line in enumerate(lines):
    
#    img, steer_angle  = Choose_image_from_camera(line) 
#    img_pp,steer_angle_pp = im_preprocess(img,steer_angle)
#    print(i)
#    print('inside' + str(img_pp.shape[0]))
#    print('inside' + str(img_pp.shape[1]))
#    if (i > 2):
#        break


# End of testing code for preprocess

# spilt Training data in 4:1 ratio or 80:20
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# generator function the parellel process batchs
def generator(samples, batch_size=32):
    num_samples=len(samples)
    Factor_Steering_Correction = 0.21

    while 1: # Loop forever so the generator never termintaes
        random.shuffle(samples)
        for offset in range (0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]            
            augmented_images = []
            augmented_measurements = []
            # Loop to process images in a batch
            for batch_sample in batch_samples:
                line = batch_sample
                imgs, steer_angles  = Choose_image_from_camera(line)       
                augmented_images.extend(imgs)
                augmented_measurements.extend(steer_angles)
            X_train = np.array(augmented_images)
            Y_train = np.array(augmented_measurements)
            #print (len(X_train))
            yield sklearn.utils.shuffle(X_train, Y_train)

batch_size=320


# compile and train the model using the generator function coded above
# only 1/3 of batch size is passed to the generator as batch size is in Preprocess 3 images are created for every 1 image.
train_generator = generator(train_samples, batch_size=int(batch_size/3))
validation_generator = generator(validation_samples, batch_size=int(batch_size/3))

# Define Simple Network
model = Sequential()

# Set up lambda Layer for data Preprocessing
model.add( Lambda(lambda x: (x/255) - 0.5 , input_shape=(160,320,3)))

# Crop the images to only road areas ignoring Skies from top of the image
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

# Implement NVIDIA Network for Autonomous Driving : End to End Driving of Self Driving Cars
model.add(Convolution2D(24, (5,5), strides=(2,2), activation="relu"))
model.add(Dropout(0.4))
model.add(Convolution2D(36, (5,5), strides=(2,2), activation="relu"))
model.add(Dropout(0.4))
model.add(Convolution2D(48, (5,5), strides=(2,2), activation="relu"))
model.add(Dropout(0.4))
model.add(Convolution2D(64, (3,3), activation="relu"))
model.add(Dropout(0.4))
model.add(Convolution2D(64, (3,3), activation="relu"))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Dense(1))

model.summary()
# Compile and fit model
# Define checkpoint and earlystopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
checkpoint = ModelCheckpoint('best_model.h5',
							 monitor='val_loss',verbose=1, save_best_only=True, 
							 save_weights_only=False, mode='auto')

# use adams optimizer and mean square error
model.compile(loss='mse',optimizer='adam')
history_obj = model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_samples)*3 / batch_size),
            validation_data=validation_generator,
            validation_steps=math.ceil(len(validation_samples)* 3 / batch_size),
            epochs=20, callbacks=[checkpoint,early_stop],verbose=1)

plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('Error_training')

# Save the model
model.save('model.h5')
