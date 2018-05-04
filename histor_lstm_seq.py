#!/afs/crc.nd.edu/x86_64_linux/t/tensorflow/1.6/gcc/python3/build/bin/python3

#An lstm sequential model built in Keras

#https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
#https://stackoverflow.com/questions/33266956/nltk-package-to-estimate-the-unigram-perplexity
#https://stackoverflow.com/questions/43932973/how-to-give-a-constant-input-to-keras
#https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/ 
#https://stackoverflow.com/questions/44960558/concatenating-embedded-inputs-to-feed-recurrent-lstm-in-keras
#https://stackoverflow.com/questions/38445982/how-to-log-keras-loss-output-to-a-file?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa	


#great for sequence to sequence:
#https://github.com/keras-team/keras/issues/2496
#https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/

#probably useful for making keras seq2seq:
#https://github.com/keras-team/keras/issues/9376
import tensorflow as tf
from tensorflow.python.keras.layers import LSTM, Dense, Input, Concatenate, Embedding
from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils import to_categorical #creates one hot encoding
import numpy as np
import pickle
#import h5py
import os

#data = ["<s> How say you? O, I should remember him: does he not hold up his head, as it were, and strut in his gait? \<s\>", "<s> He looks weird. \<s\>"] #list of sentences
#eras = [2, 4] #time periods for each
# I am not sure if this will be scalar or not, maybe we should separate into periods.

path = '/afs/crc.nd.edu/user/j/jnolan5/Private/Histor_Model/'

def generate_seq(model, tokenizer, seed, n, start_e, end_e):

	if start_e < 0 or start_e > end_e or end_e > 4:
		print("Error: values not in range")
		return 1
	in_word, result  = seed, seed

	#generate n words and print the sequence
	for i in range(n):
		era = int((float(i)/float(n))*(end_e+1 - start_e)) + start_e
		encoded = tokenizer.texts_to_sequences([in_word])[0]
		encoded = np.array(encoded)
		encoded = np.append(encoded, np.asarray(era))
		encoded = np.array([encoded])
		print("Encoded: ", encoded) 
		
		probs = model.predict(encoded)
		output = np.random.choice(np.arange(probs.size), p=probs[0]) #indices[choice][1]
		out_word = 'NULL'
		for word, index in tokenizer.word_index.items():
			if index == output:
				out_word = word
				break
		in_word = out_word
		if out_word != 'NULL':
			result += ' ' + out_word 
	print(result)
	return result

def get_data():
	data = []
	eras = []
	drct = path + 'old_english_poetry'
	for fname in os.listdir(drct):
		process_file(drct+'/'+ fname, 0, data, eras)
	drct = path + 'Middle_data/Chaucer_Geoffrey_d_1400'
	for fname in os.listdir(drct):
		process_file(drct+'/'+ fname, 1, data, eras)
	drct = path + '1500-1600_data'
	for fname in os.listdir(drct):
		process_file(drct+'/'+ fname, 2, data, eras)
	drct = path + '1700_data'
	for fname in os.listdir(drct):
		process_file(drct+'/'+ fname, 2, data, eras)
	drct = path + '1800_data'
	for fname in os.listdir(drct):
		process_file(drct+'/'+ fname, 3, data, eras)
	drct = path + '1900_data'
	for fname in os.listdir(drct):
		process_file(drct+'/'+ fname, 3, data, eras)
	drct = path + 'Modern_data'
	for fname in os.listdir(drct):
		process_file_byte(drct+'/'+ fname, 4, data, eras) #encoding issue
	return data, eras

def process_file_byte(fname, e, data, eras):
	for line in open(fname, 'rb'):
		line = str(line)
		sline = line.strip().strip('0123456789')
		if len(sline.split(' ')) > 2:
			data.append(sline)
			eras.append(e)

def process_file(fname, e, data, eras):
	for line in open(fname, 'r'):
		sline = line.strip().strip('0123456789')
		if len(sline.split(' ')) > 2:
			data.append(sline)
			eras.append(e)

data, eras = get_data()
#data is a list of sequences with a 
tok = Tokenizer(filters='!"#$%&()*+,-.:;=?@[\]^_`{|}~<>', lower=False, oov_token='NULL', num_words=50000) #should try out of vocabulary (oov_token) and limited num_words
tok.fit_on_texts(data)
vocab_size = len(tok.word_index) + 1
enc_data = tok.texts_to_sequences(data) #need to stagger all these sequences for training
cont_data = [] #continuous data
s_cont_data = [] #staggered data (continuous)
#stagger data into continuous targets

#combine eras and encoded sequence
for d, e in zip(enc_data, eras):
	for v in d:
		cont_data.append(v) 
		cont_data.append(e)

for i in range(0,len(cont_data)-2,2):
		s_cont_data.append(cont_data[i+2])

cont_data = cont_data[:-2] #remove end stop character

# add 
x = np.array(cont_data)
x = np.reshape(x, (int(x.size/2), 2))
print(x)
y = np.array(s_cont_data)

#y = to_categorical(y, num_classes=vocab_size) #no need with sparse

model = Sequential()
model.add(Embedding(vocab_size*5, 110, input_length=2))
model.add(LSTM(40)) # 40 hidden units
model.add(Dense(vocab_size, activation='softmax')) #choose next word
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #memory error with categorical_crossentropy
history = model.fit(x, y, epochs=4)
sequence = generate_seq(model, tok, 'Her', 500, 0, 4)
with open(path + 'histor_output.txt', 'w') as seq_out:
	seq_out.write(sequence)
#https://stackoverflow.com/questions/39283358/keras-how-to-record-validation-loss #might want a validation set
loss_history = np.array(history.history['loss'])
np.savetxt(path + 'loss_history.txt', loss_history, delimiter=',')
#model.save(path + 'histor_model.h5')
#https://stackoverflow.com/questions/45735070/keras-text-preprocessing-saving-tokenizer-object-to-file-for-scoring?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
