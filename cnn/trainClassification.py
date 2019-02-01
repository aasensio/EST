import numpy as np
import matplotlib.pyplot as pl
import astropy.io.fits as fits
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, AveragePooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from ipdb import set_trace as stop
import scipy as sp
import pyiacsun as ps

class trainCNN(object):

    def __init__(self):

        self.kernelNormalization = 7
        self.nFeatures = 50
        self.kernelSize = 7
        self.patchSize = 32
        self.batchSize = 32
        self.nbEpoch = 20
        self.poolSize = self.patchSize - self.kernelSize + 1
        self.nImages = 10

    def readData(self):
        print("Reading data...")
        hdu = fits.open('data/20160525-strehl.fits')
        self.strehl = hdu[0].data

        hdu = fits.open('data/20160525-r0.fits')
        self.r0 = hdu[0].data

        hdu = fits.open('data/20160525-images.fits.gz')
        self.images = hdu[0].data.T

    def extractTrainingData2(self):
        print("Extracting images...")
        nRows = 8
        self.XTrainSet = np.zeros((300,9*nRows,self.patchSize,self.patchSize))
        self.YTrainSet = np.zeros((300,9*nRows))

        self.XTestSet = np.zeros((300,9,self.patchSize,self.patchSize))
        self.YTestSet = np.zeros((300,9))

        for i in range(300):
            for j in range(9):
                left = j*32
                right = j*32+32
                for k in range(nRows):
                    top = k*32
                    bottom = k*32+32

                    self.XTrainSet[i,9*k+j,:,:] = self.images[left:right,top:bottom,i]
                    self.YTrainSet[i,9*k+j] = self.strehl[i]

                k = nRows
                top = k*32
                bottom = k*32+32
                self.XTestSet[i,j,:,:] = self.images[left:right,top:bottom,i]
                self.YTestSet[i,j] = self.strehl[i]

        mn = np.mean(self.XTrainSet, axis=(0,1))
        std = np.std(self.XTrainSet, axis=(0,1))

        self.XTrainSet = (self.XTrainSet - mn[None,None,:,:]) / std
        self.XTestSet = (self.XTestSet - mn[None,None,:,:]) / std

        self.XTrainSet = self.XTrainSet.reshape((300*9*nRows,1,self.patchSize,self.patchSize))
        self.YTrainSet = self.YTrainSet.reshape(300*9*nRows)

        tmp = np.floor(20.0*self.YTrainSet).astype('int32')
        self.YTrainSetBinary = np_utils.to_categorical(tmp, 20)

        self.XTestSet = self.XTestSet.reshape((300*9,1,self.patchSize,self.patchSize))
        self.YTestSet = self.YTestSet.reshape(300*9)

        tmp = np.floor(20.0*self.YTestSet).astype('int32')
        self.YTestSetBinary = np_utils.to_categorical(tmp, 20)

    def extractTrainingData1(self):

        # kernel = np.ones((self.kernelNormalization,self.kernelNormalization)) / self.kernelNormalization**2
        # print("Normalizing images...")
        # for i in range(300):       
        #     ps.util.progressbar(i, 300)
        #     meanMap = sp.signal.convolve2d(images[:,:,i], kernel, mode='same', boundary='symm')
        #     t1 = sp.signal.convolve2d(images[:,:,i]**2, kernel, mode='same', boundary='symm')
        #     t2 = sp.signal.convolve2d(images[:,:,i] * meanMap, kernel, mode='same', boundary='symm')
        #     t3 = sp.signal.convolve2d(meanMap**2, kernel, mode='same', boundary='symm')
        #     sigmaMap = np.sqrt(t1 + t3 - 2.0*t2)
        #     images[:,:,i] = (images[:,:,i] - meanMap) / (sigmaMap + 1e-8)

        # hdu = fits.open('data/20160525-images.fits.gz')
        # images2 = hdu[0].data.T
        # piece = images2[140-3:140+4,140-3:140+4,i]
        # mn = np.sum(piece)
        # std = np.sqrt(np.sum((piece - mn)**2))
        # print(mn, std)
        # print(meanMap[140,140] / mn, sigmaMap[140,140] / std)

        # stop()

        for i in range(300):
            maxVal = np.max(self.images[:,:,i])
            minVal = np.min(self.images[:,:,i])
            self.images[:,:,i] = 2.0 * self.images[:,:,i] / (maxVal - minVal) + (-maxVal - minVal) / (maxVal - minVal)

        print("Extracting images...")
        self.XTrainSet = np.zeros((300,self.nImages,self.patchSize,self.patchSize))
        self.YTrainSet = np.zeros((300,self.nImages))
        self.XTestSet = np.zeros((300,self.nImages,self.patchSize,self.patchSize))
        self.YTestSet = np.zeros((300,self.nImages))
        for i in range(300):
            self.XTrainSet[i,:,:,:] = extract_patches_2d(self.images[:,:,i], patch_size=(self.patchSize,self.patchSize), max_patches=self.nImages, random_state=123)
            self.YTrainSet[i,:] = self.strehl[i]

            self.XTestSet[i,:,:,:] = extract_patches_2d(self.images[:,:,i], patch_size=(self.patchSize,self.patchSize), max_patches=self.nImages, random_state=231)
            self.YTestSet[i,:] = self.strehl[i]
        
        # reorder = np.random.permutation(300)        
        # self.XTrainSet = self.XTrainSet[reorder,:]
        # self.YTrainSet = self.YTrainSet[reorder,:]

        # reorder = np.random.permutation(300)        
        # self.XTestSet = self.XTestSet[reorder,:]
        # self.YTestSet = self.YTestSet[reorder,:]

        self.XTrainSet = self.XTrainSet.reshape((300*self.nImages,1,self.patchSize,self.patchSize))
        self.YTrainSet = self.YTrainSet.reshape((300*self.nImages,1))

        self.XTestSet = self.XTestSet.reshape((300*self.nImages,1,self.patchSize,self.patchSize))
        self.YTestSet = self.YTestSet.reshape((300*self.nImages,1))

        min_max_scaler = preprocessing.MinMaxScaler()

        self.YTrainSet = min_max_scaler.fit_transform(self.YTrainSet)

# Normalization 1
        # XMean = np.mean(self.XTrainSet, axis=0)
        # XStd = np.std(self.XTrainSet, axis=0)

        # YMean = np.mean(self.YTrainSet, axis=0)
        # YStd = np.std(self.YTrainSet, axis=0)

        # self.XTrainSet -= XMean
        # self.XTrainSet /= XStd        

        # self.XTestSet -= XMean
        # self.XTestSet /= XStd

# Normalization 2
        # XMean = np.mean(self.XTrainSet, axis=(2,3))
        # XStd = np.std(self.XTrainSet, axis=(2,3))

        # self.XTrainSet -= XMean[:,:,None,None]
        # self.XTrainSet /= XStd[:,:,None,None]

        # self.XTestSet -= XMean[:,:,None,None]
        # self.XTestSet /= XStd[:,:,None,None]


    def defineCNN(self):
        print("Setting up network...")
        self.model = Sequential()
        self.model.add(Convolution2D(self.nFeatures, self.kernelSize, self.kernelSize, border_mode='valid', input_shape=(1, self.patchSize, self.patchSize), init='he_normal'))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(self.poolSize,self.poolSize)))        
        self.model.add(Flatten())
        self.model.add(Dense(800))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(800))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(20))
        self.model.add(Activation('softmax'))

        sgd = SGD(lr=0.2, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def trainCNN(self, nIterations):
        print("Training network...")
        self.metrics = self.model.fit(self.XTrainSet, self.YTrainSetBinary, batch_size=self.batchSize, nb_epoch=nIterations, validation_data=(self.XTestSet, self.YTestSetBinary), shuffle=True)
        # self.model.fit(self.XTrainSet, self.YTrainSet, batch_size=self.batchSize, nb_epoch=self.nbEpoch, validation_split=0.2)

    def testCNN(self):
        train = self.model.predict_classes(self.XTrainSet)
        test = self.model.predict(self.XTestSet)
        pl.plot(self.YTestSet, test, 'o')
        pl.plot(self.YTrainSet, train, '.')
        pl.plot(np.linspace(0.0,1.0,100), np.linspace(0.0,1.0,100))
        pl.xlim([0,1])
        pl.ylim([0,1])


out = trainCNN()
out.readData()
out.extractTrainingData2()
out.defineCNN()
 # out.defineFully()
out.trainCNN(100)
out.testCNN()