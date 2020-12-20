#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Reading Book
file = open(r"C:\Users\asus\Desktop\book\book1\book1.txt",encoding="utf8")
# variable T
T1 = file.read()

#Reading Book
file = open(r"C:\Users\asus\Desktop\book\book2\book2.txt",encoding="utf8")
# variable T
T2 = file.read()


# # punctutaions provided by python

# In[2]:


import string
print(string.punctuation)


# In[3]:


translator = str.maketrans("","",string.punctuation)
T1=T1.translate(translator)
T2=T2.translate(translator)


# In[4]:


import re


# In[5]:


# Removing acknowledgement
T1 = re.sub("The[\s\S]*CONTENTS","",T1)
# Removing the last part(Transciber's Notes)
T1 = re.sub(r"Transcriber’s[\s\S]*",r"",T1)




# In[6]:



# Removing acknowledgement
T2 = re.sub("The[\s\S]*CONTENTS","",T2)
# Removing the last part(Transciber's Notes)
T2 = re.sub(r"Transcriber’s[\s\S]*",r"",T2)


# In[7]:


# Removing Chapter Names
T1=re.sub("[A-Z]{2,}","",T1)
print(T1[:1000])


# In[8]:


# Removing Chapter Names
T2=re.sub("[A-Z]{2,}","",T2)
print(T1[:1000])


# In[9]:


T1 = T1.lower()
T2=T2.lower()


# In[10]:


# Removing Chapter Numbers
T1=re.sub("[0-9]+","",T1)
print(T1[:1000])



# In[11]:


#removing punctuations one more time
translator = str.maketrans("","",string.punctuation)
T1=T1.translate(translator)
T2=T2.translate(translator)


# In[12]:


T2=re.sub("[0-9]+","",T2)
print(T2[:1000])


# In[ ]:





# In[ ]:





# In[ ]:





# # splitting words

# In[13]:


T1=T1.split()
T2=T2.split()


# In[14]:


# FREQURNCY DISTRIBUTION OF T1 WITHOUT Lemmatisations
import nltk
freq_dist = nltk.FreqDist(T1)
import operator
sorted_T1 = sorted(freq_dist.items(),reverse=True, key=operator.itemgetter(1))
sorted_T1[:20]


# In[15]:


# FREQURNCY DISTRIBUTION OF T2 WITHOUT Lemmatisations
freq_dist = nltk.FreqDist(T2)
import operator
sorted_T2 = sorted(freq_dist.items(),reverse=True, key=operator.itemgetter(1))
sorted_T2[:20]


# In[16]:


# distribution of words with frequency for T1
m=sorted_T1[:20]
x_axis=[]
y_axis=[]
for i in range(20):
    x_axis.append(m[i][0])
    y_axis.append(m[i][1])
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.scatter(x_axis,y_axis)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of tokens for T1")
plt.show()


# In[17]:


# distribution of words with frequency for T2
m=sorted_T2[:20]
x_axis=[]
y_axis=[]
for i in range(20):
    x_axis.append(m[i][0])
    y_axis.append(m[i][1])
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.scatter(x_axis,y_axis)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of tokens for T2")
plt.show()


# In[18]:



T1 = " ".join(T1)
T2 = ' '.join(T2)
wordcloudT1=T1
wordcloudT2=T2


# In[ ]:





# In[19]:


#tokenisation
import nltk
from nltk.tokenize import word_tokenize
T1 = word_tokenize(T1)

print(T1[:100])


# In[20]:


T2 = word_tokenize (T2)
print(T2[:100])


# In[21]:


## Lemmation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Lemmatisation
lemmatizer = WordNetLemmatizer()
# forming a set of stopwords from english language
stop_words = set(stopwords.words('english'))
# creating an empty list
lemmatized_T1=[]
# traversing all the words
for word in T1:
# if the word has a length greater than or equal to 2 and is not a stopword
    if len(word) >= 2 and word not in stop_words:
    # then we append the word into the list lemmatized_T after performing␣lemmatization
    # using the lemmatize() function
        lemmatized_T1.append(lemmatizer.lemmatize(word))
# printing the list of lemmatized words
print(lemmatized_T1[:100])


# In[22]:


lemmatized_T2=[]
# traversing all the words
for word in T2:
# if the word has a length greater than or equal to 2 and is not a stopword
    if len(word) >= 2 and word not in stop_words:
    # then we append the word into the list lemmatized_T after performing␣lemmatization
    # using the lemmatize() function
        lemmatized_T2.append(lemmatizer.lemmatize(word))
# printing the list of lemmatized words
print(lemmatized_T2[:100])


# # wordcloud with stopwords

# In[23]:



T1 = " ".join(lemmatized_T1)
T2 = ' '.join(lemmatized_T2)

from wordcloud import WordCloud
wordcloud = WordCloud(width = 800, height=800,
background_color='white',stopwords={},collocations=False,
min_font_size=10).generate(wordcloudT1)
plt.figure(figsize=(8,8),facecolor=None,)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud for T1")
plt.show()


####################################
####################################
####################################


from wordcloud import WordCloud
wordcloud = WordCloud(width = 800, height=800,
background_color='white',stopwords={},collocations=False,
min_font_size=10).generate(wordcloudT2)
plt.figure(figsize=(8,8),facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud for T2")
plt.show()


# # Wordcloud without Stopwords

# In[24]:


from wordcloud import WordCloud,STOPWORDS
stopwords = set(STOPWORDS) 
wordcloud = WordCloud(width = 800, height=800,
background_color='white',stopwords=stopwords,
min_font_size=10).generate(wordcloudT1)
plt.figure(figsize=(8,8),facecolor=None,)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud for T1")
plt.show()


####################################
####################################
####################################


from wordcloud import WordCloud
wordcloud = WordCloud(width = 800, height=800,
background_color='white',stopwords=stopwords,
min_font_size=10).generate(wordcloudT2)
plt.figure(figsize=(8,8),facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud for T2")
plt.show()


# # FREQUENCY DISTRIBUTION OF WORD COUNT

# In[25]:


len_list={}
for i in range(len(lemmatized_T1)):
    if len(lemmatized_T1[i]) not in len_list:
        len_list[len(lemmatized_T1[i])] = 1
    else:
        len_list[len(lemmatized_T1[i])] += 1
keys = list(len_list.keys())
values = list(len_list.values())
plt.figure(figsize=(10,10))
plt.bar(keys,values)
plt.xlabel("Word Length")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of word length for T1")
plt.show()


###########################################
###########################################
###########################################


len_list={}
for i in range(len(lemmatized_T2)):
    if len(lemmatized_T2[i]) not in len_list:
        len_list[len(lemmatized_T2[i])] = 1
    else:
        len_list[len(lemmatized_T2[i])] += 1
keys = list(len_list.keys())
values = list(len_list.values())
plt.figure(figsize=(10,10))
plt.bar(keys,values)
plt.xlabel("Word Length")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of word length fot T2")
plt.show()


# # POS TAGGING

# In[ ]:





# In[ ]:





# In[ ]:





# In[26]:


Tagged_T1 =nltk.pos_tag(lemmatized_T1)
Tagged_T2=nltk.pos_tag(lemmatized_T2)


# In[27]:


fdistw1 = nltk.FreqDist([t for (w, t) in Tagged_T1])
fdistw2 = nltk.FreqDist([t for (w, t) in Tagged_T2])
print(Tagged_T1[:50])
print('**********************')
print(Tagged_T2[:50])


# # Keys in the corpous

# In[28]:


print("No. of tags", len(fdistw1.keys()))
fdistw1.keys()


# In[29]:


keys = (list(fdistw1.keys()))
# creating a list of the frequency of the various tokens
values = list(fdistw1.values())
# plotting a bar plot diagram of the frequency distribution
plt.figure(figsize=(10,10))
plt.bar(keys,values)
plt.xlabel("Tags")
plt.xticks(rotation=45)
plt.ylabel("Frequency")
plt.title("Frequency Distribution of tags in T1")
plt.show()

#####################################
######################################
#####################################

keys = (list(fdistw2.keys()))
# creating a list of the frequency of the various tokens
values = list(fdistw2.values())
# plotting a bar plot diagram of the frequency distribution
plt.figure(figsize=(10,10))
plt.bar(keys,values)
plt.xlabel("Tags")
plt.xticks(rotation=45)
plt.ylabel("Frequency")
plt.title("Frequency Distribution of tags in T2")
plt.show()


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




