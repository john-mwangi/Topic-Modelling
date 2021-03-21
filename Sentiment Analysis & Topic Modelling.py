#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Use Turicreate to perform sentiment analysis
# Determine what topics are evoking negative sentiment
# Evaluate model using "perplexity" - lower is better
# Group negative reviews into predetermined topics


# In[2]:


import turicreate as tc


# In[3]:


products = tc.SFrame.read_csv('products.csv')


# In[4]:


products.head()


# # Create word count (dictionary) column

# In[5]:


products['word_count'] = tc.text_analytics.count_words(text=products['review'])


# In[6]:


products.head()


# # Product exploration

# In[7]:


products['name'].show()


# In[8]:


giraffe_reviews = products[products['name']=='Vulli Sophie the Giraffe Teether']


# In[9]:


len(giraffe_reviews)


# In[10]:


giraffe_reviews['rating'].show()


# # Sentiment prediction

# In[11]:


# Exclude 3* ratings
products = products[products['rating'] != 3]


# In[12]:


# Add +ve & -ve sentiment
products['sentiment'] = products['rating'] >= 4


# In[13]:


products.head()


# # Training of the model

# In[14]:


train_data, test_data = products.random_split(fraction=0.8, seed=0)


# In[15]:


len(products), len(train_data), len(test_data)


# In[16]:


sentiment_model_LC = tc.logistic_classifier.create(dataset=train_data, features=['word_count'], target='sentiment', validation_set=test_data)
sentiment_model_TC = tc.text_classifier.create(dataset=train_data, features=['review'], drop_stop_words=True, target='sentiment', validation_set=test_data)
sentiment_model_TC_sw = tc.text_classifier.create(dataset=train_data, features=['review'], drop_stop_words=False, target='sentiment', validation_set=test_data)
# stop_words are commonly occuring words that hold no meaningful value e.g. the, if, what


# # Evaluate models using ROC

# In[17]:


sentiment_model_LC_ev = sentiment_model_LC.evaluate(dataset=test_data, metric='roc_curve')


# In[18]:


sentiment_model_TC_ev = sentiment_model_TC.evaluate(dataset=test_data, metric='roc_curve')


# In[19]:


sentiment_model_TC_sw_ev = sentiment_model_TC_sw.evaluate(dataset=test_data, metric='roc_curve')


# In[20]:


type(sentiment_model_LC_ev)


# In[21]:


len(sentiment_model_LC_ev)


# In[22]:


sentiment_model_LC_ev.keys()


# In[23]:


sentiment_model_LC_ev['roc_curve']


# In[24]:


from sklearn.metrics import auc


# In[25]:


logistic_auc = auc(x=sentiment_model_LC_ev['roc_curve']['fpr'], y=sentiment_model_LC_ev['roc_curve']['tpr'])
text_auc = auc(x=sentiment_model_TC_ev['roc_curve']['fpr'], y=sentiment_model_TC_ev['roc_curve']['tpr'])
text_sw_auc = auc(x=sentiment_model_TC_sw_ev['roc_curve']['fpr'], y=sentiment_model_TC_sw_ev['roc_curve']['tpr'])


# In[26]:


logistic_auc, text_auc, text_sw_auc


# # Plot ROC curves

# In[27]:


import matplotlib.pyplot as plt


# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


plt.plot(sentiment_model_LC_ev['roc_curve']['fpr'],
         sentiment_model_LC_ev['roc_curve']['tpr'], 
         label='logistic (AUC=%0.3f)' % logistic_auc)
plt.plot(sentiment_model_TC_ev['roc_curve']['fpr'], 
         sentiment_model_TC_ev['roc_curve']['tpr'],
         label='text (AUC=%0.3f)' % text_auc)
plt.plot(sentiment_model_TC_sw_ev['roc_curve']['fpr'], 
         sentiment_model_TC_sw_ev['roc_curve']['tpr'],
         label='text_sw (AUC=%0.3f)' % text_sw_auc)
plt.legend(loc='lower right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristics (ROC) Curve')


# In[30]:


# Conclusion - Text classifier with stop_words is most accurate


# # Predicting sentiment

# In[31]:


giraffe_reviews['predicted_sentiment'] = sentiment_model_TC_sw.predict(dataset=giraffe_reviews)


# In[32]:


giraffe_reviews.head()


# In[33]:


giraffe_reviews['predicted_prob'] = sentiment_model_TC_sw.predict(dataset=giraffe_reviews, output_type='probability')


# In[34]:


giraffe_reviews.head()


# # Sort predicted sentiment

# In[35]:


giraffe_reviews_sorted = giraffe_reviews.sort('predicted_prob', ascending=True)


# In[36]:


giraffe_reviews_sorted.head()


# In[37]:


len(giraffe_reviews_sorted)


# # Determine topics that causing negative reviews

# In[38]:


topic_model_1 = tc.topic_model.create(dataset=train_data['word_count'],
                                    num_topics=5,
                                    validation_set=test_data['word_count'])
# Build model on the larger training set, apply it on the smaller giraffe dataset


# In[39]:


giraffe_reviews['topic_1'] = topic_model_1.predict(dataset=giraffe_reviews['word_count'], output_type='assignment')


# In[40]:


giraffe_reviews.head()


# In[41]:


topic_model_1.get_topics(output_type='topic_words')


# In[42]:


# The topic model might be improved by removing these stop words


# # Removing stop words

# In[43]:


train_data['word_count_sw'] = train_data['word_count'].dict_trim_by_keys(tc.text_analytics.stop_words(), True)


# In[44]:


train_data.head()


# # Check if stop words have been removed

# In[45]:


len(train_data['word_count'][0]), len(train_data['word_count_sw'][0])


# In[46]:


train_data['word_count'][0]


# In[47]:


train_data['word_count_sw'][0]


# In[48]:


# Stop words have been removed


# In[49]:


test_data['word_count_sw'] = test_data['word_count'].dict_trim_by_keys(tc.text_analytics.stop_words(), True)


# # Second model w/o stop words

# In[50]:


topic_model_2 = tc.topic_model.create(dataset=train_data['word_count_sw'], num_topics=15, validation_set=test_data['word_count_sw'])


# In[51]:


topic_model_2.get_topics(output_type='topic_words')


# # Predict topics

# In[52]:


giraffe_reviews['topic_2'] = topic_model_2.predict(dataset=giraffe_reviews['word_count'], output_type='assignment')


# In[53]:


giraffe_reviews_sorted = giraffe_reviews.sort(key_column_names='predicted_prob', ascending=True)


# In[54]:


giraffe_reviews_sorted.head()


# # Check if topic allocation makes sense

# In[55]:


topic_model_2.get_topics(topic_ids=[13])


# In[56]:


giraffe_reviews_sorted['review'][0]


# In[57]:


giraffe_reviews_sorted['review'][7]


# In[58]:


# Topic allocation doesn't make sense. 
# Maybe it is because we're looking at the same topic across many products and too narrowed-down


# In[59]:


giraffe_reviews_sorted['word_count_sw'] = giraffe_reviews_sorted['word_count'].dict_trim_by_keys(tc.text_analytics.stop_words(), True)


# In[60]:


topic_model_3 = tc.topic_model.create(dataset=giraffe_reviews_sorted['word_count_sw'], num_topics=20)


# In[61]:


giraffe_reviews_sorted['topic_3'] = topic_model_3.predict(dataset=giraffe_reviews_sorted['word_count_sw'], output_type='assignment')


# In[62]:


giraffe_reviews_sorted.head()


# # Check topic allocation

# In[63]:


topic_model_3.get_topics(output_type='topic_words')


# In[64]:


topic_model_3.get_topics(topic_ids=[12])


# In[65]:


giraffe_reviews_sorted['review'][1]


# In[66]:


giraffe_reviews_sorted['review'][2]


# In[67]:


# The two review are now a lot more similar


# In[68]:


giraffe_reviews_sorted = giraffe_reviews_sorted.sort(key_column_names=['predicted_prob','topic_3'], ascending=True)


# In[69]:


giraffe_reviews_sorted.head()


# In[70]:


topic_model_3.get_topics(topic_ids=[9])


# In[71]:


giraffe_reviews_sorted['review'][6]


# In[72]:


giraffe_reviews_sorted['review'][8]


# # Manual topic allocation

# In[73]:


giraffe_reviews_sorted.tail()


# In[74]:


associations = tc.SFrame({'word':['rubber', 'smell', 'squeak', 'noise'],'topic': [0, 0, 1, 1]})


# In[75]:


topic_model_4 = tc.topic_model.create(associations=associations,
                                      dataset=giraffe_reviews_sorted['word_count_sw'],
                                      num_topics=20)


# In[76]:


topic_model_4.get_topics(output_type='topic_words').print_rows(num_rows=20, max_column_width=50)


# In[77]:


# Key words have been allocated to the correct topics


# In[78]:


topic_model_4.get_topics(topic_ids=[0,1,6]).print_rows(num_rows=30)


# # Predict topics

# In[79]:


giraffe_reviews_sorted['topic_4'] = topic_model_4.predict(dataset=giraffe_reviews_sorted['word_count_sw'], output_type='assignment')


# # Histogram of most common occuring negative topic

# In[80]:


giraffe_reviews_sorted.column_names


# In[81]:


giraffe_reviews_sorted['topic_4'][giraffe_reviews_sorted['predicted_sentiment']==0].show()


# In[82]:


# topic 16 has the most negative comments. We confirm if they're actually 12


# In[89]:


len(giraffe_reviews_sorted[(giraffe_reviews_sorted['predicted_sentiment']==0) & (giraffe_reviews_sorted['topic_4']==16)])


# # What is topic 16 about?

# In[90]:


topic_model_4.get_topics(topic_ids=[16])


# # Are the reviews similar?

# In[91]:


giraffe_reviews_sorted['review'][(giraffe_reviews_sorted['predicted_sentiment']==0) & (giraffe_reviews_sorted['topic_4']==16)][1]


# In[92]:


giraffe_reviews_sorted['review'][(giraffe_reviews_sorted['predicted_sentiment']==0) & (giraffe_reviews_sorted['topic_4']==16)][5]


# In[87]:


# Above is the first review


# In[93]:


giraffe_reviews_sorted['review'][(giraffe_reviews_sorted['predicted_sentiment']==0) & (giraffe_reviews_sorted['topic_4']==16)][9]


# In[ ]:


# The reviews are similar since they have a paint theme

