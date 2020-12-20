#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Reading Book
file = open(r"C:\Users\asus\Desktop\book\book1\book1.txt",encoding="utf8")
# variable T
T1 = file.read()
T11=T1

#Reading Book
file = open(r"C:\Users\asus\Desktop\book\book2\book2.txt",encoding="utf8")
# variable T
T2 = file.read()
T12=T2


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



print(T1[:10000])


# In[6]:



# Removing acknowledgement
T2 = re.sub("The[\s\S]*CONTENTS","",T2)
# Removing the last part(Transciber's Notes)
T2 = re.sub(r"Transcriber’s[\s\S]*",r"",T2)
print(T2[:10000])


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


T2=re.sub("[0-9]+","",T2)
print(T2[:1000])


# In[ ]:





# # splitting words

# In[12]:


T1=T1.split()
T2=T2.split()


# In[13]:


print(T2)
T1 = " ".join(T1)
T2 = ' '.join(T2)
wordcloudT1=T1
wordcloudT2=T2
print(T1[:1000])
print(T2[:1000])


# In[14]:


import nltk
from nltk.tokenize import word_tokenize
T1 = word_tokenize(T1)

print(T1[:100])


# In[15]:


T2 = word_tokenize (T2)
print(T2[:100])


# In[16]:



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
print(lemmatized_T1[:1000])


# In[17]:


lemmatized_T2=[]
# traversing all the words
for word in T2:
# if the word has a length greater than or equal to 2 and is not a stopword
    if len(word) >= 2 and word not in stop_words:
    # then we append the word into the list lemmatized_T after performing␣lemmatization
    # using the lemmatize() function
        lemmatized_T2.append(lemmatizer.lemmatize(word))
# printing the list of lemmatized words
print(lemmatized_T2[:1000])


# # frequency distribution plot for T1 and T2

# In[18]:


# Evaluating frequency distribution of tokens
# The nltk library function FreqDist() returns a dictionary containing key␣→value pairs where values are the frequency of
# the keys. Here keys are the Tokens.

#### for T1
freq_dist = nltk.FreqDist(lemmatized_T1)
# create a dictionary
word_dic = {}
# find the k words occuring f number of times and store in the dictionary
for i in freq_dist.keys():
    if freq_dist[i] not in word_dic:
        word_dic[freq_dist[i]] = 1
    else:
        word_dic[freq_dist[i]] += 1
# plotting a scatter plot diagram of the frequency distribution and tokens

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.scatter(word_dic.keys(),word_dic.values())
plt.xlabel("No. of Tokens")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of tokens for T1")
plt.show()

#############################################
#############################################
#############################################

### for T2
freq_dist = nltk.FreqDist(lemmatized_T2)
# create a dictionary
word_dic = {}
# find the k words occuring f number of times and store in the dictionary
for i in freq_dist.keys():
    if freq_dist[i] not in word_dic:
        word_dic[freq_dist[i]] = 1
    else:
        word_dic[freq_dist[i]] += 1
# plotting a scatter plot diagram of the frequency distribution and tokens

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.scatter(word_dic.keys(),word_dic.values())
plt.xlabel("No. of Tokens")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of tokens for T2")
plt.show()



# In[ ]:





# In[19]:


from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud(width = 800, height=800,
background_color='white',
min_font_size=10).generate(wordcloudT1)
plt.figure(figsize=(8,8),facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud for T1")
plt.show()


####################################
####################################
####################################


from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud(width = 800, height=800,
background_color='white',
min_font_size=10).generate(wordcloudT2)
plt.figure(figsize=(8,8),facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud for T2")
plt.show()


# # Removing STOPWORDS

# In[20]:


# WordCloud after removing Stopwords
wordcloud = WordCloud(width = 800, height=800,
background_color='white',
stopwords = set(STOPWORDS),
min_font_size=10).generate(wordcloudT1)
plt.figure(figsize=(8,8),facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


######################################
######################################
######################################


# WordCloud after removing Stopwords
wordcloud = WordCloud(width = 800, height=800,
background_color='white',
stopwords = set(STOPWORDS),
min_font_size=10).generate(wordcloudT2)
plt.figure(figsize=(8,8),facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# # FREQUENCY DISTRIBUTION OF WORD COUNT

# In[21]:


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





# In[22]:


Tagged_T1 =nltk.pos_tag(lemmatized_T1)
Tagged_T2=nltk.pos_tag(lemmatized_T2)


# In[23]:


fdistw1 = nltk.FreqDist([t for (w, t) in Tagged_T1])
fdistw2 = nltk.FreqDist([t for (w, t) in Tagged_T2])


# # Keys in the corpous

# In[24]:


print("No. of tags", len(fdistw1.keys()))
fdistw1.keys()


# In[25]:


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

# Round 2
# 

# In[26]:


# Seprating nouns and verbs from lemmatized words
from nltk.corpus import wordnet as wn
nouns = []
verbs = []
for word in lemmatized_T1:
    for synset in wn.synsets(word):
        if "noun" in synset.lexname() and word not in nouns:
            nouns.append(word)
        elif "verb" in synset.lexname() and word not in verbs:
            verbs.append(word)


# In[27]:


print(nouns[:100])


# In[28]:


print(verbs[:100])


# In[29]:


# Nouns
noun_dic = {}
lt = []
for noun in nouns:
    l = []
    for synset in wn.synsets(noun):
        if "noun" in synset.lexname():
            if synset.lexname()[5:] not in l:
                l.append(synset.lexname()[5:])
    noun_dic[noun] = l


# In[30]:


j = 0
for i in noun_dic:
    print(i,":", noun_dic[i])
    j +=1
    if j == 50:
        break


# In[31]:


# Verbs
verb_dic = {}
for verb in verbs:
    l = []
    for synset in wn.synsets(verb):
        if "verb" in synset.lexname():
            if synset.lexname()[5:] not in l:
                l.append(synset.lexname()[5:])
    verb_dic[verb] = l
j = 0
for i in verb_dic:
    print(i,":", verb_dic[i])
    j +=1
    if j == 50:
        break


# In[32]:


# Nouns
noun_cate_dic = {}
for i in noun_dic:
    for j in noun_dic[i]:
        if j not in noun_cate_dic:
            noun_cate_dic[j] = 1
        else:
            noun_cate_dic[j] +=1
print(noun_cate_dic)


# In[33]:


# Verbs
verb_cate_dic = {}
for i in verb_dic:
    for j in verb_dic[i]:
        if j not in verb_cate_dic:
            verb_cate_dic[j] = 1
        else:
            verb_cate_dic[j] +=1
print(verb_cate_dic)


# In[34]:


# Seprating nouns and verbs from lemmatized words
nouns2 = []
verbs2 = []
for word in lemmatized_T2:
    for synset in wn.synsets(word):
        if "noun" in synset.lexname() and word not in nouns2:
            nouns2.append(word)
        elif "verb" in synset.lexname() and word not in verbs2:
            verbs2.append(word)


# In[35]:


print(nouns2[:100])


# In[36]:


print(verbs2[:100])


# In[37]:


# Creating a dictionary of words and category of nouns they belong to.
noun_dic2 = {}
for noun in nouns2:
    d = []
    for synset in wn.synsets(noun):
        if "noun" in synset.lexname():
            if synset.lexname()[5:] not in d:
                d.append(synset.lexname()[5:])
    noun_dic2[noun] = d


# In[38]:


j = 0
for i in noun_dic2:
    print(i,":", noun_dic2[i])
    j +=1
    if j == 50:
        break


# In[39]:


# Creating a dictionary of words and category of verbs they belong to.
verb_dic2 = {}
for verb in verbs2:
    d = []
    for synset in wn.synsets(verb):
        if "verb" in synset.lexname():
            if synset.lexname()[5:] not in d:
                    d.append(synset.lexname()[5:])
    verb_dic2[verb] = d
j = 0
for i in verb_dic2:
    print(i,":", verb_dic2[i])
    j +=1
    if j == 50:
        break


# In[40]:


# Creating a dictionary of frequency of category of nouns.
noun_cate_dic2 = {}
for i in noun_dic2:
    for j in noun_dic2[i]:
        if j not in noun_cate_dic2:
            noun_cate_dic2[j] = 1
        else:
            noun_cate_dic2[j] +=1
print(noun_cate_dic2)


# In[41]:


# Creating a dictionary of frequency of category of verbs.
verb_cate_dic2 = {}
for i in verb_dic2:
    for j in verb_dic2[i]:
        if j not in verb_cate_dic2:
            verb_cate_dic2[j] = 1
        else:
            verb_cate_dic2[j] +=1
print(verb_cate_dic2)


# In[42]:


# Bar graph of Noun Category and Frequency in Book1
plt.figure()
plt.bar(noun_cate_dic.keys(),noun_cate_dic.values())
plt.title('Noun Category Vs Frequency(for Book 1)')
plt.xlabel('Noun Category')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


# In[43]:


# Bar graph of Verb Category and Frequency in Book1
plt.figure()
plt.bar(verb_cate_dic.keys(),verb_cate_dic.values())
plt.title('Verb Category Vs Frequency(for Book 1)')
plt.xlabel('Verb Category')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


# We are using bar graph instead of histogram as the x-axis of the data has discrete values

# In[44]:


# Bar graph of Noun Category and Frequency in Book2
plt.figure()
plt.bar(noun_cate_dic2.keys(),noun_cate_dic2.values())
plt.title('Noun Category Vs Frequency(for Book 2)')
plt.xlabel('Noun Category')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


# In[45]:


# Bar graph of Verb Category and Frequency in Book2
plt.figure()
plt.bar(verb_cate_dic2.keys(),verb_cate_dic2.values())
plt.title('Verb Category Vs Frequency(for Book 2)')
plt.xlabel('Verb Category')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


# PART 2
# 

# Entity recongition and Classifiaction

# In[46]:


from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk import ne_chunk
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report


# In[47]:


from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


# In[48]:


tagged_T2 = nltk.word_tokenize(T12)
tagged_T2 = nltk.pos_tag(tagged_T2)
print(tagged_T2[:100])


# Using nltk function ne_chunk for chunking the text and named entity recognition
# 

# In[49]:


results = ne_chunk(tagged_T2)
print(results[:100])


# Creating a parse tree from the tagged text

# In[50]:


pattern = 'NP: {<DT>?<JJ>*<NN>}'
cp = nltk.RegexpParser(pattern)
cs = cp.parse(tagged_T2)
print(cs[:100])


# Performing IOB tagging using nltk

# In[51]:


from nltk.chunk import tree2conlltags
from pprint import pprint
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged[:50])


# Representing the IOB tags

# In[52]:


j = 0
for word, pos, ner in iob_tagged:
    print(word, pos, ner)
    j +=1
    if j == 50:
        break


# converting the text to tokens, does IOB tagging and entity recongnition and
# classification using spacy
# 

# In[53]:


T12= nlp(T12)


# Below are the IOB tags and entity types of the text
# 

# In[54]:


for X in T12[6350:6360]:
    print(X, X.ent_iob_, X.ent_type_)


# labels of the entity types, labeled by spacy

# In[55]:


labels = [x.label_ for x in T12.ents]


# In[56]:


Counter(labels)


# using nltk for pos tagging and tokenizing the text which is required later for chunking and
# IOB tagging

# In[57]:


tagged_S2 = nltk.word_tokenize(T11)
tagged_S2 = nltk.pos_tag(tagged_S2)
tagged_S2[:100]


# nltk function ne_chunk for chunking the text and named entity recognition
# 

# In[58]:


results2 = ne_chunk(tagged_S2)
print(results2[:100])


# Creating a parse tree from the tagged text

# In[59]:


pattern = 'NP: {<DT>?<JJ>*<NN>}'
cp2 = nltk.RegexpParser(pattern)
cs2 = cp2.parse(tagged_S2)
print(cs2[:100])


# Performing IOB tagging using nltk

# In[60]:


from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
iob_tagged2 = tree2conlltags(cs2)
pprint(iob_tagged2[:50])


# Representing the IOB tags using print function

# In[61]:


j = 0
for word, pos, ner in iob_tagged2:
    print(word, pos, ner)
    j +=1
    if j == 10:
        break


# nlp function of spacy converts the text to tokens, does IOB tagging and entity recongnition and
# classification

# In[62]:


S2 = nlp(T11)


# In[63]:


for X in S2[6350:6360]:
    print(X, X.ent_iob_, X.ent_type_)


# We extract all the labels of the entity types, labeled by spacy

# In[64]:


labels2 = [x.label_ for x in S2.ents]


# Counter function of spacy counts all the types of labels in the text

# In[65]:


Counter(labels2)


# # Performanace Measures

# Since we only require the entity types specified in fig. 22.1 of chapter 22 we only extract those
# in our predicted entity list. Below only a part of the text (from 6350 to 6750 character) is shown
# because we use the same for finding the F-score

# In[66]:


entity_pred = []
for X in T12[6350:6750]:
    if X.ent_type_ == "GPE" or X.ent_type_ == "PERSON" or X.ent_type_ == "ORG"or X.ent_type_ == "FAC" or X.ent_type_ == "LOC":
        entity_pred.append('B-'+X.ent_type_)
    elif X.ent_type_=="":
        entity_pred.append('O')
print(entity_pred)


# In[67]:


entity_pred = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'B-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],['B-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],['B-PERSON', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O'], ['B-FAC', 'I-FAC', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O','O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'I-PERSON','O', 'O'], ['B-PERSON', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O']]


# We now create a entity type list of the actual values in the paragraph which we tagged manually

# In[68]:


entity_true = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'B-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],['B-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],['B-PERSON','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-LOC', 'I-LOC', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O'], ['B-PERSON', 'O', 'O','O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON','I-PERSON', 'O', 'O'], ['B-PERSON', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O']]


# F-score between actual entity types and predicted entity types

# In[69]:


f1_score(entity_true, entity_pred)


# Below is the accuracy score for the same

# In[70]:


accuracy_score(entity_true, entity_pred)


# BOOK 2
# 
# 
# Since we only require the entity types specified in fig. 22.1 of chapter 22 we only extract those
# in our predicted entity list. Below only a part of the text (from 7050 to 7500 character) is shown
# because we use the same for finding the F-score

# In[71]:


entity_pred2 = []
for X in S2[7050:7500]:
    if X.ent_type_ == "GPE" or X.ent_type_ == "PERSON" or X.ent_type_ == "ORG" or X.ent_type_ == "FAC" or X.ent_type_ == "LOC":
        entity_pred2.append(X.ent_type_)
    elif X.ent_type_=="":
        entity_pred2.append('O')
print(entity_pred2)


# In[72]:


entity_pred2 =['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'PERSON', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'GPE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'GPE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORG', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'GPE', 'GPE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O']


# In[73]:


entity_pred2 = [['O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'O','O'], ['B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'I-PERSON', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-LOC', 'O', 'O'], ['B-ORG', 'I-ORG','I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],['B-GPE', 'O', 'O', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-LOC', 'O', 'O'], ['B-PERSON','I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O'], ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],['B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O']]


# In[74]:


entity_true2 = [['O', 'O', 'O', 'O', 'B-PERSON', 'I-PERSON', 'I-PERSON','I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O','O', 'O'], ['B-PERSON', 'I-PERSON', 'I-PERSON', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'I-PERSON', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-LOC', 'O', 'O'],['B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON','I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O'], ['B-GPE', 'O', 'O', 'O', 'O'], ['B-PERSON','I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],['B-LOC', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'I-PERSON','I-PERSON', 'I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'I-PERSON','I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O']]


# Finding the F-score between actual entity types and predicted entity types

# In[75]:


f1_score(entity_true2, entity_pred2)


# Below is the accuracy score for the same

# In[76]:


accuracy_score(entity_true2, entity_pred2)


# Extracting Relationship between Entities

# In[77]:


TEXTS = [
"""Secretary Stanton, just arrived from the bedside of Mr. Seward, asked Surgeon-General Barnes what was Mr. Lincol
n's condition. "I fear, Mr. Stanton, that there is no hope." "O, no, general; no, no;" and the man, of all others, ap
parently strange to tears, sank down beside the bed, the hot, bitter evidences of an awful sorrow trickling through h
is fingers to the floor. Senator Sumner sat on the opposite side of the bed, holding one of the President's hands in
his own, and sobbing with kindred grief. Secretary Welles stood at the foot of the bed, his face hidden, his frame s
haken with emotion. General Halleck, Attorney-General Speed, Postmaster-General Dennison, M. B. Field, Assistant Secr
etary of the Treasury, Judge Otto, General Meigs, and others, visited the chamber at times, and then retired. Mrs. Li
ncoln--but there is no need to speak of her. Mrs. Senator Dixon soon arrived, and remained with her through the nigh
t. All through the night, while the horror-stricken crowds outside swept and gathered along the streets, while the mi
litary and police were patrolling and weaving a cordon around the city; while men were arming and asking each other,
"What victim next?" while the telegraph was sending the news from city to city over the continent, and while the two
assassins were speeding unharmed upon fleet horses far away--his chosen friends watched about the death-bed of the hi
ghest of the nation. Occasionally Dr. Gurley, pastor of the church where Mr. Lincoln habitually attended, knelt down
in prayer. Occasionally Mrs. Lincoln and her sons, entered, to find no hope and to go back to ceaseless weeping. Mem
bers of the cabinet, senators, representatives, generals, and others, took turns at the bedside. Chief-Justice Chase
remained until a late hour, and returned in the morning."""
]


# Below are the named entities extracted using the model

# In[78]:


ner_model = spacy.load(r'en_core_web_sm')
def ner_text():
    doc = ner_model(TEXTS[0])
    for entity in doc.ents:
        print(entity.label_,' ',entity.text)
ner_text()


# Lets see the relation b/w PERSON

# In[79]:


def filter_spans(spans):
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result


# In[80]:


def main(ner_model):
    nlp = ner_model
    print("Processing %d texts" % len(TEXTS))
    for text in TEXTS:
        doc = nlp(text)
        relations = extract_per_relations(doc)
        for r1, r2 in relations:
            print("{:<10}\t{}\t{}".format(r1.text, r2.ent_type_, r2.text))


# In[81]:


def extract_per_relations(doc):
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)
    relations = []
    for per in filter(lambda w: w.ent_type_ == "PERSON", doc):
        if per.dep_ in ("attr", "dobj"):
            subject = [w for w in per.head.lefts if w.dep_ == "nsubj"]
            if subject:
                subject = subject[0]
                relations.append((subject, per))
        elif per.dep_ == "pobj" and per.head.dep_ == "prep":
            relations.append((per.head.head, per))
    return relations


# In[82]:


main(ner_model)


# In[83]:


TEXTS = ["""The last man, who sits on the extreme right of the prisoners, is Mr. Sam. Arnold. He is, perhaps, the bes
t looking of the prisoners, and theleast implicated. He has a solid, pleasant face; has been a rebel soldier, foolish
ly committed himself to Booth, with perhaps no intention to do a crime, recanted in pen and ink, and was made a natio
nal character. Had he recanted by word of mouth he might have saved himself unpleasant dreams. This shows everybody t
he absurdity of writing what they can so easily say. The best thing Arnold ever wrote was his letter to Booth refusin
g to engage in murder. Yet this recantation is more in evidence against than then his original purpose."""
]


# Text 2

# In[84]:


ner_text()


# Lets see the relations b/w PERSON's

# In[85]:


main(ner_model)


# In[ ]:




