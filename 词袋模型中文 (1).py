#!/usr/bin/env python
# coding: utf-8

# In[19]:


pip install wordcloud


# In[78]:


import matplotlib
from matplotlib import font_manager

a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])

for i in a:
    print(i)


# In[77]:


#


# In[69]:


# fc-list :lang=zh family


# In[44]:


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# In[45]:


text_ana=pd.read_csv('text_analysis_weibo.csv')
text_ana=text_ana.drop(['Unnamed: 0'], axis=1)
text_ana


# In[46]:


text=list(text_ana)
text


# In[47]:


import jieba

def word_segmentation(text):
    words = jieba.lcut(text)
    return ' '.join(words)

text_ana['标题/微博内容'] = text_ana['标题/微博内容'].iloc[:100]\
                                .astype(str).apply(word_segmentation)
text_ana['标题/微博内容'].head()


# In[48]:


text = list(text_ana['标题/微博内容'].dropna())
text


# In[49]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(text)

print('词袋特征矩阵的形状:', X.shape)


# In[50]:


print('特征词列表:', vectorizer.get_feature_names_out())


# In[51]:


list(X.toarray())


# In[52]:


feature_words = vectorizer.get_feature_names_out()

word_freq = dict(zip(feature_words, X.sum(axis=0).A1))


# In[53]:


len(word_freq)


# In[54]:


sorted(word_freq.items(), key=lambda x: x[1], reverse=True)


# In[79]:


wordcloud = WordCloud(
   # plt.rcParams['font.sans-serif'] = ['Heiti SC']
 font_path='STHeitiTC-Medium-01.ttf', 
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




