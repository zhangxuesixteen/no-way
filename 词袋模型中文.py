#!/usr/bin/env python
# coding: utf-8

# In[19]:


pip install wordcloud


# In[17]:


import pandas as pd
from wordcloud import WordCloud


# In[18]:


text_ana=pd.read_csv('text_analysis_weibo.csv')
text_ana=text_ana.drop(['Unnamed: 0'], axis=1)
text_ana


# In[19]:


text=list(text_ana)
text


# In[20]:


import jieba

def word_segmentation(text):
    words = jieba.lcut(text)
    return ' '.join(words)

text_ana['标题/微博内容'] = text_ana['标题/微博内容'].iloc[:100]\
                                .astype(str).apply(word_segmentation)
text_ana['标题/微博内容'].head()


# In[21]:


text = list(text_ana['标题/微博内容'].dropna())
text


# In[22]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(text)

print('词袋特征矩阵的形状:', X.shape)


# In[23]:


print('特征词列表:', vectorizer.get_feature_names_out())


# In[24]:


list(X.toarray())


# In[25]:


feature_words = vectorizer.get_feature_names_out()

word_freq = dict(zip(feature_words, X.sum(axis=0).A1))


# In[26]:


len(word_freq)


# In[27]:


sorted(word_freq.items(), key=lambda x: x[1], reverse=True)


# In[28]:


wordcloud = WordCloud(
    font_path='simhei.ttf', 
    background_color='white',
    width=1200,
    height=800,
    max_words=200,
    max_font_size=100
).generate_from_frequencies(word_freq)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:




