
# coding: utf-8

# In[2]:


import time

# import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#text libraries
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.preprocessing import text,sequence


# In[4]:


import gensim
from keras.preprocessing import text
from fasttext import load_model
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D


# In[5]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# copy test id column for later submission
result = test[['id']].copy() 
# show first 3 rows of the training set to get a first impression about the data
print(train.head(3))


# In[9]:


replacement_patterns = [
 (r'won\'t', 'will not'),
 (r'don\'t', 'do not'),
 (r'does\'t', 'does not'),
 (r'did\'t', 'did not'),
 (r'can\'t', 'cannot'),
 (r'i\'m', 'I am'),
 (r'I\'m', 'I am'),
 (r'ain\'t', 'is not'),
 (r'ain\'t', 'is not'),
 (r'(\w+)\'ll', '\g<1> will'),
 (r'(\w+)n\'t', '\g<1> not'),
 (r'(\w+)\'ve', '\g<1> have'),
 (r'(\w+)\'s', '\g<1> is'),
 (r'(\w+)\'re', '\g<1> are'),
 (r'(\w+)\'d', '\g<1> would')
]
class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
         self.patterns = [(re.compile(regex), repl) for (regex, repl) in
         patterns]
     
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
             s = re.sub(pattern, repl, s)
        return s


# In[11]:


regExp = RegexpReplacer()
category_processed = []
stopwords = nltk.corpus.stopwords.words('english')
a_list = []
for i in range(train.shape[0]):
    s = regExp.replace(train['comment_text'][i])
    nstr = re.sub(r'[$|\n|:|â|€|â€|=|/|ãƒ|©|‡|‰|«|ƒ|â†|Â|~|_|™|¦|]',r' ',s)
    a_list.append(nstr)
dataTrain = pd.DataFrame({'id': train['id'],'comment_text': a_list})


# In[13]:


def text_to_words(raw_text, remove_stopwords=False):
    #Remove non-letters, but including numbers

    words = raw_text.split()
    if remove_stopwords:
        stops = set(stopwords.words("english")) # In Python, searching a set is much faster than searching
        meaningful_words = [w for w in words if not w in stops] # Remove stop words
        words = meaningful_words
    return words 

sentences_train = dataTrain['comment_text'].apply(text_to_words, remove_stopwords=False)
#sentences_test = trest['comment_text'].apply(text_to_words, remove_stopwords=False)
# show first three arrays as sample
print(sentences_train[4:11])


# In[14]:


# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words
# Initialize and train the model (this will take some time)
model = word2vec.Word2Vec(sentences_train, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)
model.init_sims(replace=True) # marks the end of training to speed up the use of the model


# In[15]:


model.doesnt_match('idiot beauty nice flower'.split())


# In[16]:


ftModel = model


# In[17]:


ftModel.most_similar("mother")


# In[18]:


X_train = dataTrain["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values


max_features = 30000
maxlen = 100
embed_size = 100

tokenizer = text.Tokenizer(num_words=max_features,lower= False)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


# In[19]:


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))


# In[20]:


for word, i in word_index.items():
    if i >= max_features or len(word) <2: continue
    try:
        embedding_vector = d[word]
    except:
        continue
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[21]:


#ROC-AUC is the metric used in the Kaggle competition, we will be using the same for evaluating our model
rnnAccuracy =0
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            rnnAccuracy =score
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# In[22]:


model = get_model()


batch_size = 32
epochs = 2

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc], verbose=2)


# In[60]:


submission = pd.read_csv('sample_submission.csv')


# In[61]:


y_pred = model.predict(x_test, batch_size=1024)


# In[56]:


submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission.csv', index=False)


# In[24]:


x_test.shape




#Testing our own examples to intuitively understand the effectiveness of the classifier
singleTest = pd.DataFrame([ "THIS IS FUCKING AMAZING", "YOU FUCKING DARE DO" "I think you're an idiot" , "I don't think so" ,"I hate you", "I HATE YOU"])
sinTest = tokenizer.texts_to_sequences(singleTest[0])
sin_test = sequence.pad_sequences(sinTest, maxlen=maxlen)
sin_pred = model.predict(sin_test)
sin_pred_pd = pd.DataFrame(sin_pred, columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
sin_pred_pd["text"] = singleTest[0]
sin_pred_pd = sin_pred_pd[[ "text","toxic","severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
sin_pred_pd






