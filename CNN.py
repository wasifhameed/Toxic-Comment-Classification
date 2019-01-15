
# coding: utf-8

# In[24]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU,Conv1D,MaxPooling1D
from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import gc
from sklearn.model_selection import train_test_split
from keras.models import load_model


# In[25]:


from keras.preprocessing import text,sequence


# In[26]:


train = pd.read_csv('train.csv')
submit = pd.read_csv('test.csv')


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(train, train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]], test_size = 0.10, random_state = 42)


# In[28]:


list_sentences_train = X_train["comment_text"]
list_sentences_test = X_test["comment_text"]
list_sentences_submit = submit["comment_text"]


# In[29]:


max_features = 20000
tokenizer = Tokenizer(num_words=max_features,char_level=True)


# In[30]:


tokenizer.fit_on_texts(list(list_sentences_train))


# In[31]:


list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_sentences_test = tokenizer.texts_to_sequences(list_sentences_test)
list_tokenized_submit = tokenizer.texts_to_sequences(list_sentences_submit)


# In[32]:


maxlen = 500
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_sentences_test, maxlen=maxlen)
X_sub = pad_sequences(list_tokenized_submit, maxlen=maxlen)


# In[33]:


inp = Input(shape=(maxlen, ))


# In[34]:


embed_size = 240
x = Embedding(len(tokenizer.word_index)+1, embed_size)(inp)


# In[35]:


x = Conv1D(filters=100,kernel_size=4,padding='same', activation='relu')(x)


# In[36]:


x=MaxPooling1D(pool_size=4)(x)


# In[37]:


x = Bidirectional(GRU(60, return_sequences=True,name='lstm_layer',dropout=0.2,recurrent_dropout=0.2))(x)


# In[38]:


x = GlobalMaxPool1D()(x)


# In[39]:


x = Dense(50, activation="relu")(x)


# In[40]:


x = Dropout(0.2)(x)
x = Dense(6, activation="sigmoid")(x)


# In[41]:


model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                 metrics=['accuracy'])


# In[40]:


model.summary()


# In[41]:


batch_size = 32
epochs = 6

hist = model.fit(X_t,y_train, batch_size=batch_size, epochs=epochs,validation_data=(X_te,y_test))


# In[42]:


import h5py


# In[43]:


model.save('savedCNN.h5')


# In[44]:


model


# In[45]:


batch_size = 32
y_submit = model.predict(X_sub,batch_size=batch_size,verbose=1)


# In[49]:


y_submit


# In[47]:


submit_template = pd.read_csv('sample_submission.csv', header = 0)


# In[ ]:


y_submit[np.isnan(y_submit)]=0
sample_submission = submit_template
sample_submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_submit
sample_submission.to_csv('submission.csv', index=False)


# In[12]:


max_features=100000
maxlen=150
embed_size=300


# In[19]:


tok=text.Tokenizer(num_words=100000,lower=True)
tok.fit_on_texts(list(X_train)+list(X_test))
X_train=tok.texts_to_sequences(X_train)
X_test=tok.texts_to_sequences(X_test)
x_train=sequence.pad_sequences(X_train,maxlen=maxlen)
x_test=sequence.pad_sequences(X_test,maxlen=maxlen)


# In[20]:


X_test


# In[15]:


x_train

