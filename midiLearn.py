from scipy.io.wavfile import read as wavread
import numpy as np
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Reshape
from keras.layers import Input
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam # I do not know which to use
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping # first saves the best model, second loggs the training, third stops the training of the result does not improve
from keras.models import load_model # to load saved model.
from generateWavs import getFilteredDataList
from keras.layers.merge import Concatenate, Add
from keras.layers import concatenate
import os.path
import librosa.core
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn.preprocessing
from tqdm import tqdm

#parameters
#lr = 0.01 # learning rate -> valid acc 0.9881
#lr = 0.0075 # -> valid acc 9860
#separate network 98.32
#merged network: 0.9961
lr = 0.01
emptyCombinations = np.zeros((1,77),dtype=np.int)
wavFileNames  = {"nylon_20.wav": "combinationMatrix.npy",
                  "160465__matucha__computer-noise-desktop-quadcore-2009.wav": emptyCombinations.copy(),
                  "199896__qubodup__office-ambience.wav": emptyCombinations.copy(),
                  "343755__brokenphono__scary-guitar.wav" : emptyCombinations.copy(),
                  "43400__noisecollector__themostannoyingmeetingever.wav": emptyCombinations.copy() } # wav filename - combMatrix (string, ndarray) pairs.



def train_model(cqts, combinations, lr):  # trains the model and returns the saved file's filename. (could also return the model, which is not necessarily identical to the saved one.)
	model_name = 'model.hd5'
	log_dir = 'TB_log'
	callbacks = []
	callbacks.append(TensorBoard(log_dir=log_dir)) # logs into the TB_log directory
	callbacks.append(ModelCheckpoint(filepath=model_name,   verbose=1, save_best_only=True, period=1)) # saves the model if the result is improved at the current epoch
	callbacks.append(EarlyStopping(patience = 3,verbose = 1))
#	model = Sequential()
#	model.add(LSTM(combinations.shape[2],activation = "sigmoid", return_sequences = True,  input_shape = (cqts.shape[1],cqts.shape[2]))) # sigmoid activation, since the output is scaled between 0 and 1.	
#	model.add(LSTM(combinations.shape[2],activation = "sigmoid", return_sequences = True,  input_shape = (cqts.shape[1],cqts.shape[2]))) # sigmoid activation, since the output is scaled between 0 and 1.	
#	model.add(TimeDistributed(Dense(combinations.shape[2],activation="sigmoid")))
#	model.add(TimeDistributed(Dense(combinations.shape[2],activation="sigmoid")))

	i = Input(shape = (cqts.shape[1],cqts.shape[2]))
	left0  = TimeDistributed(Dense(cqts.shape[2], activation = "sigmoid" ))(i)
	left1 = LSTM(10,activation="sigmoid",return_sequences = True)(left0)
	left2 = TimeDistributed(Dense(4,activation="sigmoid"))(left1)
	left3 = TimeDistributed(Dense(4,activation = "softmax"))(left2)

	right0 = TimeDistributed(Dense(cqts.shape[2],activation="sigmoid"))(i)
	right1 = LSTM(combinations.shape[2]-4,activation = "sigmoid", return_sequences = True)(right0)
	right2 = TimeDistributed(Dense(combinations.shape[2]-4,activation = "sigmoid"))(right1)
	right3 = TimeDistributed(Dense(combinations.shape[2]-4,activation = "sigmoid"))(right2)

	o = concatenate([left3,right3])
	model = Model(inputs= i, outputs = o)
	print("Model Summary: \n" + str(model.summary()))
	optimizer = RMSprop(lr=lr) #RMSprop(lr = lr)
	model.compile(optimizer=optimizer, loss='binary_crossentropy',  metrics=['accuracy'])
	model.fit(cqts, combinations, epochs=150, batch_size=120, shuffle=False,validation_split = 0.2, callbacks = callbacks)

	#not that dat is already shuffled in the function processWav.
	return model_name # model is saved with the ModelCheckpoint callback.


def cutWavByNotes(x, noteDurationInBeat, bpm, sampleRate): # x: wav data in numpy array
	bps = bpm / 60  # beat per sec
	noteDurationinSamples = (noteDurationInBeat * sampleRate) / bps  # note duration in samples
	# Checking if wav format is okay.
	numberOfNotes = (len(x)) / float(noteDurationinSamples)
	numberOfNotesInt = int(np.floor(numberOfNotes))
	noteDurationinSamplesInt = int(noteDurationinSamples)
	if len(x) != numberOfNotesInt * noteDurationinSamplesInt: # x array length is not a multiple of a note length
		print("In cutWavByNotes function, the length of the data is ", len(x), " samples.",
                         "The length of a note is determined to be ", noteDurationinSamples, " samples",
                         "Which means that the total number of notes in the data is ", numberOfNotes,
                         "One of them is not an integer number.\n Cutting wav data from length " , len(x) ,
						  " to length " ,numberOfNotesInt*noteDurationinSamplesInt )
		x = x[:numberOfNotesInt*noteDurationinSamplesInt]
	x = x.reshape(numberOfNotesInt, noteDurationinSamplesInt)  # reshapes: every row is a note
	return x

# Reads a (filtered) wav file, and returns training data (cqt transormed samples and output vectors)
#Convention: In the wav file, each note of the note indicated in the combMatrix lasts for noteDurationInBeat beats.

def processWav(x,sampleRate, combMatrix = None):
	# used variables
	sampleLength = 512  # in samples ( 44100 is not a multiple of 512...)
	# this depends on sound rendering/recording bpm
	noteDurationInBeat = 4  # sound duration in beats ( each sound is followed by silence of the same length.)
	bpm = 120
	# calculated variables:
	bps = bpm / 60  # beat per sec
	noteDurationinSamples = noteDurationInBeat * sampleRate / bps # note duration in samples


	if combMatrix is None:
		if os.path.isfile("combinationMatrix.npy"):
			combMatrix = np.load("combinationMatrix.npy")
		elif isinstance(combMatrix, str):
			if os.path.isfile(combMatrix):
				combMatrix = np.load(combMatrix)
			else:
				raise ValueError("ProcessWav searches for the file ", combMatrix, " but it is not found.")
		elif not isinstance(combMatrix, np.ndarray):
			raise ValueError("Process Wav needs the combination matrix ( generated by createMIDI), but file is not found.")


	xCut = cutWavByNotes(x,noteDurationInBeat,bpm,sampleRate)#each note in a row
	if combMatrix.shape[0] ==1: # only one row is given:  apply this combination to all notes in the dataset.
		combMatrix = np.repeat(combMatrix,xCut.shape[0],axis=0)


	samples, combMatrix = shuffle(xCut, combMatrix, random_state=13002)  # shuffles matricies.


	samplesFromNote = int(np.floor(noteDurationinSamples / sampleLength)) # we are creating this many samples from one note.
	# repeating notes in the comb matrix
	combMatrix = np.tile(combMatrix,samplesFromNote).reshape(-1,combMatrix.shape[1]) # repeats each row as many times as many samples we create from a single note.

	#oneHot encodes combination matrix
	print("OneHot encoding Combinations Matrix")
	#maxCombMatrix = np.max(combMatrix,0)
	#maxNotesTogether = maxCombMatrix[0] # first col is  the number of notes played together.
	#minCombMatrix = np.min(combMatrix,0)
	#minMidiNumber = np.min(minCombMatrix[1:])
	#maxMidiNumber = np.max(maxCombMatrix[1:])
	#TODO: make dynamic but it should work with all zeros...
	# each row of the onehot encoded matrix will look like this:
	# [----]  [-----------------------] First section represents the number of notes played together, going from 0 [1000] to maxNotesTogether. (e.g. in case of 4: 00001]
	# second section is a range of midi Notes. Played midi Notes will have value 1.
	midiRange = 77-4
	noteNumberRange = 4
	#DEBUG
	print(combMatrix[0,:])
	combOneHot = np.zeros((combMatrix.shape[0],noteNumberRange+midiRange))
	for i in tqdm(range(combMatrix.shape[0])):
		notes = combMatrix[i,0] # first col tells how many notes are played.
		combOneHot[i,notes] = 1 # indicates how many notes are played
		for j in range(notes):
			note = combMatrix[i,j+1]
			combOneHot[i,noteNumberRange + note - minMidiNumber] = 1 # indicates which note.
  
		
	#preparing samples
	samples = samples[:,:samplesFromNote*sampleLength] # cutting the end. (44100 is not a multiple of 512)

	samples = samples.flatten() # flat array
	#Shuffle is applied before reshape(sampleLengthm-1), samples from the same notes are following each other, but
	#note combinations are shuffled.
	print("Calculating CQTs")
	cqt = librosa.core.cqt(samples.astype(np.float),sampleRate,sampleLength)[:,:-1]
	cqt = librosa.core.logamplitude(cqt).transpose()
	cqts = sklearn.preprocessing.normalize(cqt,axis=1)

	# Creating sequences.
	sequenceLength = 3
	cutFromEnd = cqts.shape[0] %  (samplesFromNote * sequenceLength)
	if cutFromEnd>0:
		cqts = cqts[:-cutFromEnd,:]
		combOneHot = combOneHot[:-cutFromEnd,:]
	cqts = cqts.reshape(-1,samplesFromNote * sequenceLength, cqts.shape[1]) # 3 note in a sequence
	combOneHot = combOneHot.reshape(-1,samplesFromNote * sequenceLength, combOneHot.shape[1])
	return (cqts, combOneHot)


def get_data(fileName, combMatrix = None):
	#Reads data from disk if exists. If not, generates data from waw from wav file with name  fileName.
	dataFileName = fileName + "_trainData.npz"
	if os.path.isfile(dataFileName):
		print('Loading data from file ' + dataFileName)
		data = np.load(dataFileName)
		cqt_transform = data['cqt_transform']
		combinations = data['combinations']
	else:
		cqt_transform = None
		combinations = None
		print('Generating training data from waw.')
		(note_sample_list, sample_rate) = getFilteredDataList(fileName)
		while note_sample_list:
			print('Processing new x data. List size is ' + str(len(note_sample_list)))
			x =note_sample_list.pop()
			(cqt_transform_this, combinations_this) = processWav(x, sample_rate, combMatrix)
			if cqt_transform_this.shape[0] != combinations_this.shape[0]:
				raise ValueError("processWav returned with two matrices with the difference sizes. cqt shape is " + str(cqt_transform_this.shape) +
							", combinations shape: " + str(combinations_this.shape))
			if cqt_transform is None:
				cqt_transform = cqt_transform_this
				combinations = combinations_this
			else:
				cqt_transform = np.vstack([cqt_transform,cqt_transform_this])
				combinations = np.vstack([combinations, combinations_this])
		print("Data generation finished, saving it to disk.")
		np.savez(dataFileName,cqt_transform = cqt_transform, combinations = combinations)
		print('Data saved to disk for next training.')
	return (cqt_transform, combinations)

np.random.seed(13002) # for reproductivity. (fyi: '13' is 'B', '0' is 'O' and '2' is 'Z')
cqt_transform = []
combinations = []
for file, combMatrix in wavFileNames.items():
	print("processing file " , file)
	(cqt_transform_this, combinations_this) = get_data(file, combMatrix)
	if len(cqt_transform) ==0:
		cqt_transform = cqt_transform_this
		combinations = combinations_this
	else:
		cqt_transform = np.vstack([cqt_transform,cqt_transform_this])
		combinations = np.vstack([combinations,combinations_this])


# Splitting the dataset to train (inc. validation) and test set.
cqt_transform_train, cqt_transform_test, combinations_train, combinations_test = train_test_split(cqt_transform,combinations, test_size = 0.15, random_state = 13002  )

#Train  ( chu - chu )
print("Training with :\ntrain_data shape: " , cqt_transform_train.shape, " label shape: ",combinations_train.shape)

trained_model_path = train_model(cqt_transform_train,combinations_train, lr)

#Load saved model and test on the test data
print("Reloadig the model from the hard drive and testing it using the test dataset.")
model = load_model(trained_model_path)
test_accuracy = model.evaluate(cqt_transform_test,combinations_test)[1] # returns the loss and accuracy in this order.
print("\nTest accuracy: " + str(test_accuracy))

print("See the training log by executing $ tensorboard --logdir='.'")
