



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

data_path = './data'
train = pd.read_csv('%s/train.csv.gz' % data_path, parse_dates=['Date'])
test  = pd.read_csv('%s/test.csv.gz' % data_path,  parse_dates=['Date'])



train.head().T




test.head().T




print 'Train min/max date: %s / %s' % (train.Date.min().date(), train.Date.max().date())
print 'Test  min/max date: %s / %s' % ( test.Date.min().date(),  test.Date.max().date())
print ''
print 'Number of days in train: %d' % ((train.Date.max() - train.Date.min()).days + 1)
print 'Number of days in test:  %d' % (( test.Date.max() -  test.Date.min()).days + 1)
print ''
print 'Train shape: %d rows' % train.shape[0]
print 'Test shape: %d rows'  % test.shape[0]





test.Date.value_counts()





train.Date.value_counts().head(10)




test_nunique = test.nunique()
test_nunique





import itertools

# This function looks for a combination of elements 
# with product of 639360 
def find_prod(data):
    # combinations of not more than 5 features
    for n in range(1, 5):
        # iterate through all combinations
        for c in itertools.combinations(range(len(data)), n):
            if data[list(c)].prod() == 639360:
                print test_nunique.index[c]
                return
    print 'Nothing found'

    
find_prod(test_nunique.values)


# Hmm, nothing found! The problem is that some features are tied, and the number of their combinations does not equal to product of individual unique number of elements. For example it does not make sense to create all possible combinations of *DestinationUrl* and *AdGroupId* as *DestinationUrl* belong to exactly one *AdGroupId*.




test.groupby('DestinationUrl').AdGroupId.nunique()


# So, now let's try to find ID differently. Let's try to find a list of columns, such that threre is exazctly $639360$ unique combinations of their values **in the test set** (not overall). So, we want to find `columns`, such that:



test[columns].drop_duplicates().shape[0]  == 639360


# We could do it with a similar loop.




import itertools

def find_ncombinations(data):
    # combinations of not more than 5 features
    for n in range(1, 5):
        for c in itertools.combinations(range(data.shape[1]), n):
            print c
            columns = test.columns[list(c)]
            if test[columns].drop_duplicates().shape[0] == 639360:
                print columns
                return
    print 'Nothing found'

    
find_ncombinations(test)


# But it will take forever to compute. So it is easier to find the combination manually.

# So after some time of trials and errors I figured out, that the four features *KeywordId, AdGroupId, Device, Slot* form the index. The number of unique rows is exactly *639360* as we wanted to find.




columns = ['KeywordId', 'AdGroupId', 'Device', 'Slot']
test[columns].drop_duplicates().shape


# Looks reasonable. For each *AdGroupId* there is a **distinct set** of possible *KeywordId's*, but *Device* and *Slot* variants are the same for each ad. And the target is to predict what will be the daily cost for using different *KeywordId's*, *Device* type, *Slot* type to advertise ads from *AdGroups*.

# ### Train

# To this end, we found how test set was constructed, but what about the train set? Let us plot something, probably we will find it out. 




import seaborn as sns
sns.set(palette='pastel')
sns.set(font_scale=2)





# from absolute dates to relative
train['date_diff'] =  (train.Date - train.Date.min()).dt.days





# group by the index, that we've found
g= train.groupby(['KeywordId', 'AdGroupId', 'Device', 'Slot'])

# and for each index show average relative date versus 
# the number of rows with that index
plt.figure(figsize=(12,12))
plt.scatter(g.date_diff.mean(),g.size(),edgecolor = 'none',alpha = 0.2, s=20, c='b')
plt.xlabel('Group mean relative date')
plt.ylabel('Group size')
plt.title('Train');


# Looks interesting, isn't it? That is something we need to explain! How the same plot looks for the test set?



# from absolute dates to relative
test['date_diff'] =  (test.Date - test.Date.min()).dt.days





# group by the index, that we've found
g= test.groupby(['KeywordId', 'AdGroupId', 'Device', 'Slot'])

# and for each index show average relative date versus 
# the number of rows with that index
plt.figure(figsize=(12,12))
plt.scatter(g.date_diff.mean(),g.size(),edgecolor = 'none',alpha = 0.2, s=20, c='b')
plt.xlabel('Group mean relative date')
plt.ylabel('Group size')
plt.ylim(-2, 30)
plt.title('Test');


# Just a dot! 
# 
# Now let's think, what we actually plotted? We grouped the data by the ID that we've found previously and we plotted average *Date* in the group versus the size of each group. We found that ID is an aggregation index -- so for each date the *Cost* is aggreagated for each possible index. So group size shows for how many days we have *Const* information for each ID and mean relative date shows some information about these days.    
# 
# For test set it is expectable that both average date and the size of the groups are the same for each group: the size of each group is $14$ (as we have $14$ test days) and mean date is $6.5$, because for each group (index) we have $14$ different days, and $\frac{0 + 1 + \dots + 13}{14} = 6.5$.
# 
# And now we can explain everything for the train set. Look at the top of the triangle: for those points (groups) we have *Cost* information for all the days in the train period, while on the sides we see groups, for which we have very few rows.
# 
# But why for some groups we have smaller number of rows, than number of days? Let's look at the *Impressions* column.




train.Impressions.value_counts()


# We never have $0$ value in *Imressions* column. But in reality, of course, some ads with some combination of keyword, slot, device were never shown. So this looks like a nice explanation for the data: in the train set we **only**  have information about ads (IDs, groups) which were shown at least once. And for the test set, we, of course, want to predict *Cost* **for every** possible ID. 

# What it means for competitors, is that if one would just fit a model on the train set as is, the predictions for the test set will be biased by a lot. The predictions will be much higher than they should be, as we are only given a specific subset of rows as `train.csv` file. 

# So, before modeling we should first extend the trainset and inject rows with `0` impressions. Such change will make train set very similar to the test set and the models will generalize nicely.  
