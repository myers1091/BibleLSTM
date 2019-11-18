import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pickle
import sys

#load the network weights
modelshape = pickle.load(open("modelshape.p",'rb'))
filename = 'weights/weights-improvement-20-1.3382.hdf5'
model = Sequential()
model.add(LSTM(256, input_shape=(modelshape["Xshape1"], modelshape["Xshape2"])))
model.add(Dropout(0.2))
model.add(Dense(modelshape["Yshape"], activation='softmax'))
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_char = dict((i,c) for i, c in enumerate(modelshape["chars"]))

#input pattern, can make it anything
dataX = modelshape["dataX"]
n_vocab = modelshape["n_vocab"]
start = numpy.random.randint(0,len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print ("\nDone.")