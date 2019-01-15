
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


toxic = pd.read_csv('train.csv')


# In[ ]:


toxic.head()


# In[ ]:


toxic.shape


# In[ ]:


toxic = toxic.set_index('id')


# In[ ]:


toxic.head()


# In[3]:


toxic['text length'] = toxic['comment_text'].apply(len)


# In[4]:


sns.distplot(a=toxic['text length'],bins=30)


# In[ ]:


toxic.hist(column='text length', by='toxic', bins=50,figsize=(12,4))


# In[ ]:


toxic.hist(column='text length', by='severe_toxic', bins=50,figsize=(12,4))


# In[6]:


import string
from nltk.corpus import stopwords


# In[ ]:


toxic['comment_text'].fillna("unknown", inplace=True)


# In[ ]:


toxic.head()


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


# In[9]:


tfidf_vec = TfidfVectorizer(max_df=0.7,stop_words='english')


# In[10]:


X = toxic['comment_text']
y = toxic['toxic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_toxic = LogisticRegression()
log_toxic.fit(X_train_vec,y_train)

predictions = log_toxic.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[11]:


X = toxic['comment_text']
y = toxic['severe_toxic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_stoxic = LogisticRegression()
log_stoxic.fit(X_train_vec,y_train)

predictions = log_stoxic.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[12]:


X = toxic['comment_text']
y = toxic['obscene']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_obscene = LogisticRegression()
log_obscene.fit(X_train_vec,y_train)

predictions = log_obscene.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[13]:


X = toxic['comment_text']
y = toxic['threat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_threat = LogisticRegression()
log_threat.fit(X_train_vec,y_train)

predictions = log_threat.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[14]:


X = toxic['comment_text']
y = toxic['insult']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_insult = LogisticRegression()
log_insult.fit(X_train_vec,y_train)

predictions = log_insult.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[15]:


X = toxic['comment_text']
y = toxic['identity_hate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_ihate = LogisticRegression()
log_ihate.fit(X_train_vec,y_train)

predictions = log_ihate.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[16]:


test = pd.read_csv('test.csv')
test.head()


# In[17]:


test1 = test['comment_text']
test1_vec = tfidf_vec.transform(test1)


# In[18]:


prob_toxic = log_toxic.predict_proba(test1_vec)
prob_stoxic = log_stoxic.predict_proba(test1_vec)
prob_obscene = log_obscene.predict_proba(test1_vec)
prob_threat = log_obscene.predict_proba(test1_vec)
prob_insult = log_insult.predict_proba(test1_vec)
prob_ihate = log_ihate.predict_proba(test1_vec)


# In[19]:


df1 = pd.DataFrame(prob_toxic[:,1],columns={'toxic'})
df2 = pd.DataFrame(prob_stoxic[:,1],columns={'severe_toxic'})
df3 = pd.DataFrame(prob_obscene[:,1],columns={'obscene'})
df4 = pd.DataFrame(prob_threat[:,1],columns={'threat'})
df5 = pd.DataFrame(prob_insult[:,1],columns={'insult'})
df6 = pd.DataFrame(prob_ihate[:,1],columns={'identity_hate'})


# In[20]:


dfConcat = pd.concat([test['id'],df1,df2,df3,df4,df5,df6],axis=1)


# In[21]:


dfConcat.head()


# In[22]:


dfConcat.set_index('id',inplace=True)


# In[23]:


dfConcat.to_csv(path_or_buf='submission.csv')

