
# coding: utf-8

# In[2]:


pd.set_option('max_columns', 100)


# # Load the data

# In[3]:


train = pd.read_csv('./train.csv')
train.head()


# # Build a quick baseline

# In[4]:


from sklearn.ensemble import RandomForestClassifier

# Create a copy to work with
X = train.copy()

# Save and drop labels
y = train.y
X = X.drop('y', axis=1)

# fill NANs 
X = X.fillna(-999)

# Label encoder
for c in train.columns[train.dtypes == 'object']:
    X[c] = X[c].factorize()[0]
    
rf = RandomForestClassifier()
rf.fit(X,y)


# In[5]:


plt.plot(rf.feature_importances_)
plt.xticks(np.arange(X.shape[1]), X.columns.tolist(), rotation=90);


# There is something interesting about `x8`.

# In[6]:


# we see it was standard scaled, most likely, if we concat train and test, we will get exact mean=1, and std 1 
print 'Mean:', train.x8.mean()
print 'std:', train.x8.std()


# In[7]:


# And we see that it has a lot of repeated values
train.x8.value_counts().head(15)


# In[8]:


# It's very hard to work with scaled feature, so let's try to scale them back
# Let's first take a look at difference between neighbouring values in x8

x8_unique = train.x8.unique()
x8_unique_sorted = np.sort(x8_unique)
                           
np.diff(x8_unique_sorted)


# In[9]:


# The most of the diffs are 0.04332159! 
# The data is scaled, so we don't know what was the diff value for the original feature
# But let's assume it was 1.0
# Let's devide all the numbers by 0.04332159 to get the right scaling
# note, that feature will still have zero mean

np.diff(x8_unique_sorted/0.04332159)


# In[10]:


(train.x8/0.04332159).head(10)


# In[11]:


# Ok, now we see .102468 in every value
# this looks like a part of a mean that was subtracted during standard scaling
# If we subtract it, the values become almost integers
(train.x8/0.04332159 - .102468).head(10)


# In[12]:


# let's round them 
x8_int = (train.x8/0.04332159 - .102468).round()
x8_int.head(10)


# In[13]:


# Ok, what's next? In fact it is not obvious how to find shift parameter, 
# and how to understand what the data this feature actually store
# But ...


# In[14]:


x8_int.value_counts()


# In[ ]:


# do you see this -1968? Doesn't it look like a year? ... So my hypothesis is that this feature is a year of birth! 
# Maybe it was a textbox where users enter their year of birth, and someone entered 0000 instead
# The hypothesis looks plausible, isn't it?


# In[22]:


(x8_int + 1968.0).value_counts().sort_index()


# In[23]:


# After the competition ended the organisers told it was really a year of birth

