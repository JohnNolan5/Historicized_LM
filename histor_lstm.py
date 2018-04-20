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

from keras.layers import LSTM, Dense, Input, Concatenate, Embedding
from keras.models import Sequential

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical #creates one hot encoding
import tensorflow as tf
import numpy as np
import os

#data = ["<s> How say you? O, I should remember him: does he not hold up his head, as it were, and strut in his gait? \<s\>", "<s> He looks weird. \<s\>"] #list of sentences
#eras = [2, 4] #time periods for each
# I am not sure if this will be scalar or not, maybe we should separate into periods.

def get_data():
	data = []
	eras = []
	drct = 'old_english_poetry'
	for fname in os.listdir(drct):
		process_file(drct+'/'+ fname, 0, data, eras)
	drct = 'Middle_data/Chaucer_Geoffrey_d_1400'
	for fname in os.listdir(drct):
		process_file(drct+'/'+ fname, 1, data, eras)
	drct = 'Middle_data/middle_english'
	for fname in os.listdir(drct):
		process_file(drct+'/'+ fname, 1, data, eras)
	drct = '1500-1600_data'
	for fname in os.listdir(drct):
		process_file(drct+'/'+ fname, 2, data, eras)
	drct = '1700_data'
	for fname in os.listdir(drct):
		process_file(drct+'/'+ fname, 2, data, eras)
	drct = '1800_data'
	for fname in os.listdir(drct):
		process_file(drct+'/'+ fname, 3, data, eras)
	drct = '1900_data'
	for fname in os.listdir(drct):
		process_file(drct+'/'+ fname, 3, data, eras)
	drct = 'Modern_data'
	for fname in os.listdir(drct):
		process_file_byte(drct+'/'+ fname, 4, data, eras) #encoding issue
	return data, eras

def process_file_byte(fname, e, data, eras):
	for line in open(fname, 'rb'):
		line = str(line)
		sline = line.strip().strip('0123456789')
		if len(sline.split(' ')) > 2:
			data.append('<s> '+sline+' </s>')
			eras.append(e)

def process_file(fname, e, data, eras):
	for line in open(fname, 'r'):
		sline = line.strip().strip('0123456789')
		if len(sline.split(' ')) > 2:
			data.append('<s> '+sline+' </s>')
			eras.append(e)

data, eras = get_data()
#data is a list of sequences with a 
tok = Tokenizer(filters='!"#$%&()*+,-.:;=?@[\]^_`{|}~', lower=False) #should try out of vocabulary (oov_token) and limited num_words
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
y = np.array(s_cont_data)
print('x')
print(x)
print('y')
print(y)

y = to_categorical(y, num_classes=vocab_size)

model = Sequential()
model.add(Embedding(vocab_size*5, 90, input_length=2))
model.add(LSTM(50)) # 50 hidden units
model.add(Dense(vocab_size, activation='softmax')) #choose next word
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x, y, epochs=10)
model.save('histor_model.h5')
loss_history = np.array(history.history['loss'])
np.savetxt('loss_history.txt', loss_history, delimiter=',')
