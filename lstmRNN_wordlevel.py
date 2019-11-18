import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from string import punctuation
import pickle
import string

def depunct(n):
    return n.strip(punctuation)
def split(word): 
    return [char for char in word]  
def spacer(stringyboi):
    stringyboi = stringyboi.replace(","," , ")
    stringyboi = stringyboi.replace("."," . ")
    stringyboi = stringyboi.replace("\""," \" ")
    stringyboi = stringyboi.replace(":"," : ")
    stringyboi = stringyboi.replace(";"," ; ")
    stringyboi = stringyboi.replace("?"," ? ")
    stringyboi = stringyboi.replace("!"," ! ")
    stringyboi = stringyboi.replace("  "," ")
    return stringyboi

#Import file and convert to lowercase
filename = 'Data/The Book of Psalms.txt'
raw_text = open(filename,'r',encoding='utf-8').read()
raw_text = raw_text.lower()
raw_text = spacer(raw_text)
word_list = raw_text.split()
words = word_list
#words = list(map(depunct,word_list))
#[word.strip(punctuation) for word in word_list]
#[word.strip(string.punctuation) for word in word_list]

words_only = sorted(list(dict.fromkeys(words)))
chars = words_only + [" "]

#map unique chars to integers
#NOTE: could instead divide by word (regex to select up to \s or \.\, etc.)
char_to_int = dict((c,i)for i,c in enumerate(chars))

n_chars = len(words)
n_vocab = len(chars)
print("Total Characters: ",n_chars)
print("Total Vocab: ",n_vocab)

#Split dataset into sequences
#Use 1st sequence to predict second sequence
#1st attempt is to use seq_length of 100

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = words[i:i + seq_length]
    seq_out = words[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
#print("Total Patterns: ", n_patterns)
#Reshape X to be [samples,time steps, features]
X = numpy.reshape(dataX,(n_patterns, seq_length, 1))
#normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
model_shape = {"dataX":dataX,"Xshape1":X.shape[1],"Xshape2":X.shape[2],"Yshape":y.shape[1],"chars":chars,"n_vocab":n_vocab}
pickle.dump( model_shape, open('modelshape_words.p','wb'))

#define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#define the checkpoint
filepath = "weights/weights-words-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]

#fit the model
model.fit(X, y, epochs = 50, batch_size =64, callbacks = callbacks_list)