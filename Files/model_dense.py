import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils.np_utils import to_categorical
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob
import scipy.io
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Activation


def unet(pretrained_weights = None,input_size = (256,256,1)):
            inputs = Input(input_size)

            '''
            unet with crop(because padding = valid) 
            conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(inputs)
            print "conv1 shape:",conv1.shape
            conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
            print "conv1 shape:",conv1.shape
            crop1 = Cropping2D(cropping=((90,90),(90,90)))(conv1)
            print "crop1 shape:",crop1.shape
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            print "pool1 shape:",pool1.shape
            conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
            print "conv2 shape:",conv2.shape
            conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv2)
            print "conv2 shape:",conv2.shape
            crop2 = Cropping2D(cropping=((41,41),(41,41)))(conv2)
            print "crop2 shape:",crop2.shape
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            print "pool2 shape:",pool2.shape
            conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
            print "conv3 shape:",conv3.shape
            conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv3)
            print "conv3 shape:",conv3.shape
            crop3 = Cropping2D(cropping=((16,17),(16,17)))(conv3)
            print "crop3 shape:",crop3.shape
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            print "pool3 shape:",pool3.shape
            conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool3)
            conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv4)
            drop4 = Dropout(0.5)(conv4)
            crop4 = Cropping2D(cropping=((4,4),(4,4)))(drop4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
            conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool4)
            conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv5)
            drop5 = Dropout(0.5)(conv5)
            up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
            merge6 = merge([crop4,up6], mode = 'concat', concat_axis = 3)
            conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge6)
            conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv6)
            up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
            merge7 = merge([crop3,up7], mode = 'concat', concat_axis = 3)
            conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge7)
            conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv7)
            up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
            merge8 = merge([crop2,up8], mode = 'concat', concat_axis = 3)
            conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge8)
            conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv8)
            up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
            merge9 = merge([crop1,up9], mode = 'concat', concat_axis = 3)
            conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge9)
            conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
            conv9 = Conv2D(2, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
            '''

            conv1_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
            BatchNorm1_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv1_1)
            ReLU1_1 = Activation('relu')(BatchNorm1_1)
            conv1_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU1_1)
            drop1_2 = Dropout(0)(conv1_2)
            Merge1 = concatenate([conv1_1,drop1_2], axis = 3)
            pool1 = MaxPooling2D(pool_size=(2, 2))(Merge1)

            conv2_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
            BatchNorm2_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv2_1)
            ReLU2_1 = Activation('relu')(BatchNorm2_1)
            conv2_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU2_1)
            drop2_2 = Dropout(0)(conv2_2)
            Merge2 = concatenate([conv2_1,drop2_2], axis = 3)
            pool2 = MaxPooling2D(pool_size=(2, 2))(Merge2)

            conv3_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
            BatchNorm3_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv3_1)
            ReLU3_1 = Activation('relu')(BatchNorm3_1)
            conv3_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU3_1)
            drop3_2 = Dropout(0)(conv3_2)
            Merge3 = concatenate([conv3_1,drop3_2],axis = 3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(Merge3)

            conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
            BatchNorm4_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv4_1)
            ReLU4_1 = Activation('relu')(BatchNorm4_1)
            conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU4_1)
            drop4_2 = Dropout(0)(conv4_2)
            Merge4 = concatenate([conv4_1,drop4_2], axis = 3)
            drop4 = Dropout(0.5)(Merge4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

            conv5_1 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
            BatchNorm5_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv5_1)
            ReLU5_1 = Activation('relu')(BatchNorm5_1)
            conv5_2 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU5_1)
            drop5_2 = Dropout(0)(conv5_2)
            Merge5 = concatenate([conv5_1,drop5_2], axis = 3)
            drop5 = Dropout(0.5)(Merge5)

            up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
            merge6 = concatenate([drop4,up6], axis = 3)
            conv6_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
            BatchNorm6_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv6_1)
            ReLU6_1 = Activation('relu')(BatchNorm6_1)
            conv6_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU6_1)
            drop6_2 = Dropout(0)(conv6_2)
            Merge6 = concatenate([conv6_1,drop6_2], axis = 3)

            up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge6))
            merge7 = concatenate([Merge3,up7],  axis = 3)
            conv7_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
            BatchNorm7_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv7_1)
            ReLU7_1 = Activation('relu')(BatchNorm7_1)
            conv7_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU7_1)
            drop7_2 = Dropout(0)(conv7_2)
            Merge7 = concatenate([conv7_1,drop7_2], axis = 3)

            up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge7))
            merge8 = concatenate([Merge2,up8], axis = 3)
            conv8_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
            BatchNorm8_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv8_1)
            ReLU8_1 = Activation('relu')(BatchNorm8_1)
            conv8_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU8_1)
            drop8_2 = Dropout(0)(conv8_2)
            Merge8 = concatenate([conv8_1,drop8_2], axis = 3)

            up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge8))
            merge9 = concatenate([Merge1,up9],  axis = 3)
            conv9_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
            BatchNorm9_1 = BatchNormalization(axis=3, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv9_1)
            ReLU9_1 = Activation('relu')(BatchNorm9_1)
            conv9_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ReLU9_1)
            drop9_2 = Dropout(0)(conv9_2)
            Merge9 = concatenate([conv9_1,drop9_2], axis = 3)

            conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge9)
            conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)#sigmoid
                    #conv10 = Conv2D(1, 1, activation = 'softmax')(conv9)#sigmoid

            model = Model(input = inputs, output = conv10)

            model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
                    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
            if(pretrained_weights):
                model.load_weights(pretrained_weights)

            return model

            