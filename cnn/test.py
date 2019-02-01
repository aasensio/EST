import numpy as np
import matplotlib.pyplot as pl
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, AveragePooling2D, BatchNormalization
from keras.optimizers import SGD
from ipdb import set_trace as stop

class trainCNN(object):

    def __init__(self):

        self.X = np.random.randn(200,1)
        self.Y = 1.2*self.X**2 + 0.5

    def defineCNN(self):
        print("Setting up network...")
        self.model = Sequential()
        self.model.add(Dense(40, input_shape=(1,)))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(1))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mse', optimizer='RMSprop')

    def trainCNN(self, nIterations):
        print("Training network...")
        self.metrics = self.model.fit(self.X, self.Y, batch_size=20, nb_epoch=nIterations, validation_split=0.2, shuffle=False)
        # self.model.fit(self.XTrainSet, self.YTrainSet, batch_size=self.batchSize, nb_epoch=self.nbEpoch, validation_split=0.2)

    def testCNN(self):
        train = self.model.predict(self.X)
        pl.plot(self.X, self.Y, '.')
        pl.plot(self.X, train, 'x')


out = trainCNN()
out.defineCNN()
# out.defineFully()
out.trainCNN(1)
out.testCNN()