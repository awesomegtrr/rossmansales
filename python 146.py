############# IMPORTING, CONFIG SETTINGS AND LOADING DATA ##################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import pylab
import csv
import datetime
import math
import re
import time
import random
import os

from pandas.tseries.offsets import *
from operator import *
from sklearn.cross_validation import train_test_split

# %matplotlib inline

# plt.style.use('ggplot') # Good looking plots

np.set_printoptions(precision=4, threshold=10000, linewidth=100, edgeitems=999, suppress=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 100)
pd.set_option('expand_frame_repr', False)
pd.set_option('precision', 6)

start_time = time.time()


def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

	
def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe

seed = 42

nrows = None
rounds = 1500 #set number of rounds for algorithm to take for both training and testing

df_train = pd.read_csv('data sets/trainSeason.csv', 
                       nrows=nrows,
                       parse_dates=['Date'],
                       date_parser=(lambda dt: pd.to_datetime(dt, format='%Y-%m-%d')))

nrows = nrows

df_submit = pd.read_csv('data sets/test.csv', 
                        nrows=nrows,
                        parse_dates=['Date'],
                        date_parser=(lambda dt: pd.to_datetime(dt, format='%Y-%m-%d')))

### Setting a variable to easily distinguish train (1) from submit (0) set
df_train['Set'] = 1
df_submit['Set'] = 0

#############################################################################################

### Combine train and test set
frames = [df_train, df_submit]
df = pd.concat(frames)

df.info()

# features to play with 
features_x = ['Store', 'Date', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'StateHoliday', 'SeasonalFactor']
features_y = ['SalesLog']

### Remove rows where store is open, but no sales.
df = df.loc[~((df['Open'] == 1) & (df['Sales'] == 0))]

df.loc[df['Set'] == 1, 'SalesLog'] = np.log1p(df.loc[df['Set'] == 1]['Sales']) # = np.log(df['Sales'] + 1)


df['StateHoliday'] = df['StateHoliday'].astype('category').cat.codes #Make state holidays into categories


#create own categories by splitting the dates into day, week, month, year and day of the year
var_name = 'Date'

df[var_name + 'Day'] = pd.Index(df[var_name]).day
df[var_name + 'Week'] = pd.Index(df[var_name]).week
df[var_name + 'Month'] = pd.Index(df[var_name]).month
df[var_name + 'Year'] = pd.Index(df[var_name]).year
df[var_name + 'DayOfYear'] = pd.Index(df[var_name]).dayofyear


#fill up blanks with zeroes
df[var_name + 'Day'] = df[var_name + 'Day'].fillna(0)
df[var_name + 'Week'] = df[var_name + 'Week'].fillna(0)
df[var_name + 'Month'] = df[var_name + 'Month'].fillna(0)
df[var_name + 'Year'] = df[var_name + 'Year'].fillna(0)
df[var_name + 'DayOfYear'] = df[var_name + 'DayOfYear'].fillna(0)

features_x.remove(var_name)
features_x.append(var_name + 'Day')
features_x.append(var_name + 'Week')
features_x.append(var_name + 'Month')
features_x.append(var_name + 'Year')
features_x.append(var_name + 'DayOfYear')

#turn date into float
df['DateInt'] = df['Date'].astype(np.int64)



#read store data
df_store = pd.read_csv('data sets/store.csv', 
                       nrows=nrows)

df_store.info()

### Similar to state holidays, we convert Storetype and Assortment into categories
df_store['StoreType'] = df_store['StoreType'].astype('category').cat.codes
df_store['Assortment'] = df_store['Assortment'].astype('category').cat.codes


### This is to convert competition open's year and month to float
def convertCompetitionOpen(df):
    try:
        date = '{}-{}'.format(int(df['CompetitionOpenSinceYear']), int(df['CompetitionOpenSinceMonth']))
        return pd.to_datetime(date)
    except:
        return np.nan

df_store['CompetitionOpenInt'] = df_store.apply(lambda df: convertCompetitionOpen(df), axis=1).astype(np.int64)

### This is to convert promotion's year and month to float
def convertPromo2(df):
    try:
        date = '{}{}1'.format(int(df['Promo2SinceYear']), int(df['Promo2SinceWeek']))
        return pd.to_datetime(date, format='%Y%W%w')
    except:
        return np.nan

df_store['Promo2SinceFloat'] = df_store.apply(lambda df: convertPromo2(df), axis=1).astype(np.int64)

#split the months in each column into intervals
s = df_store['PromoInterval'].str.split(',').apply(pd.Series, 1)
s.columns = ['PromoInterval0', 'PromoInterval1', 'PromoInterval2', 'PromoInterval3']
df_store = df_store.join(s)

#for turning dates into an integer value
def monthToNum(date):
    return{
            'Jan' : 1,
            'Feb' : 2,
            'Mar' : 3,
            'Apr' : 4,
            'May' : 5,
            'Jun' : 6,
            'Jul' : 7,
            'Aug' : 8,
            'Sept' : 9, 
            'Oct' : 10,
            'Nov' : 11,
            'Dec' : 12
    }[date]

df_store['PromoInterval0'] = df_store['PromoInterval0'].map(lambda x: monthToNum(x) if str(x) != 'nan' else np.nan)
df_store['PromoInterval1'] = df_store['PromoInterval1'].map(lambda x: monthToNum(x) if str(x) != 'nan' else np.nan)
df_store['PromoInterval2'] = df_store['PromoInterval2'].map(lambda x: monthToNum(x) if str(x) != 'nan' else np.nan)
df_store['PromoInterval3'] = df_store['PromoInterval3'].map(lambda x: monthToNum(x) if str(x) != 'nan' else np.nan)


### Created Features were not helping
# PromoInterval1, PromoInterval2, PromoInterval3
del df_store['PromoInterval']

store_features = ['Store', 'StoreType', 'Assortment', 
                  'CompetitionDistance', 'CompetitionOpenInt',
                  'PromoInterval0']


features_x = list(set(features_x + store_features))

df = pd.merge(df, df_store[store_features], how='left', on=['Store'])

## Can do this at the start instead
### Convert every NAN to -1
for feature in features_x:
    df[feature] = df[feature].fillna(-1)

# list_stores_to_check = [105,163,172,364,378,523,589,663,676,681,700,708,730,764,837,845,861,882,969,986]

# plt.rcParams["figure.figsize"] = [20,len(list_stores_to_check)*5]

# j = 1
# for i in list_stores_to_check:
#     stor = i

#     # Normal sales
#     X1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1)]
#     y1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1)]['Sales']

#     Xt = df.loc[(df['Store'] == stor)]
    
#     plt.subplot(len(list_stores_to_check),1,j)
#     plt.plot(X1['DateInt'], y1, '-')
#     plt.minorticks_on()
#     plt.grid(True, which='both')
#     plt.title(i)
#     j += 1


# list_stores_to_check = [192,263,500,797,815,825]

# plt.rcParams["figure.figsize"] = [20,len(list_stores_to_check)*5]

# j = 1
# for i in list_stores_to_check:
#     stor = i

#     # Normal sales
#     X1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1)]
#     y1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1)]['Sales']

#     Xt = df.loc[(df['Store'] == stor)]
    
#     plt.subplot(len(list_stores_to_check),1,j)
#     plt.plot(X1['DateInt'], y1, '-')
#     plt.minorticks_on()
#     plt.grid(True, which='both')
#     plt.title(i)
#     j += 1

# list_stores_to_check = [274,524,709,1029]

# plt.rcParams["figure.figsize"] = [20,len(list_stores_to_check)*5]

# j = 1
# for i in list_stores_to_check:
#     stor = i

#     # Normal sales
#     X1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor)]
#     y1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor)]['Sales']

#     Xt = df.loc[(df['Store'] == stor)]
    
#     plt.subplot(len(list_stores_to_check),1,j)
#     plt.plot(X1['DateInt'], y1, '.')
#     plt.minorticks_on()
#     plt.grid(True, which='both')
#     plt.title(i)
#     j += 1

# list_stores_to_check = [274,524,709,1029]

# plt.rcParams["figure.figsize"] = [20,len(list_stores_to_check)*5]

# j = 1
# for i in list_stores_to_check:
#     stor = i

#     # Normal sales
#     X1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1)]
#     y1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1)]['Sales']

#     Xt = df.loc[(df['Store'] == stor)]
    
#     plt.subplot(len(list_stores_to_check),1,j)
#     plt.plot(X1['DateInt'], y1, '-')
#     plt.minorticks_on()
#     plt.grid(True, which='both')
#     plt.title(i)
#     j += 1

# list_stores_to_check = [299,453,530,732,931]

# plt.rcParams["figure.figsize"] = [20,len(list_stores_to_check)*5]

# j = 1
# for i in list_stores_to_check:
#     stor = i

#     # Normal sales
#     X1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1)]
#     y1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1)]['Sales']

#     Xt = df.loc[(df['Store'] == stor)]
    
#     plt.subplot(len(list_stores_to_check),1,j)
#     plt.plot(X1['DateInt'], y1, '-')
#     plt.minorticks_on()
#     plt.grid(True, which='both')
#     plt.title(i)
#     j += 1

store_dates_to_remove = {   105:1.368e18, 163:1.368e18,
                            172:1.366e18, 364:1.37e18,
                            378:1.39e18, 523:1.39e18,
                            589:1.37e18, 663:1.39e18,
                            676:1.366e18, 681:1.37e18,
                            700:1.373e18, 708:1.368e18,
                            709:1.423e18, 730:1.39e18,
                            764:1.368e18, 837:1.396e18,
                            845:1.368e18, 861:1.368e18,
                            882:1.368e18, 969:1.366e18,
                            986:1.368e18, 192:1.421e18,
                            263:1.421e18, 500:1.421e18,
                            797:1.421e18, 815:1.421e18,
                            825:1.421e18}

for key,value in store_dates_to_remove.iteritems():
    df.loc[(df['Store'] == key) & (df['DateInt'] < value), 'Delete'] = True

# list_stores_to_check = [105,163,172,364,378,523,589,663,676,681,700,708,730,764,837,845,861,882,969,986]

# plt.rcParams["figure.figsize"] = [20,len(list_stores_to_check)*5]

# j = 1
# for i in list_stores_to_check:
#     stor = i

#     # Normal sales
#     X1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1) & (df['Delete'] == True)]
#     y1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1) & (df['Delete'] == True)]['Sales']
    
#     X2 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1) & (df['Delete'] != True)]
#     y2 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1) & (df['Delete'] != True)]['Sales']

#     Xt = df.loc[(df['Store'] == stor)]
    
#     plt.subplot(len(list_stores_to_check),1,j)
#     plt.plot(X1['DateInt'], y1, 'r-')
#     plt.plot(X2['DateInt'], y2, '-')
#     plt.minorticks_on()
#     plt.grid(True, which='both')
#     plt.title(i)
#     j += 1

### Delete the data where sales in the first period is much different from the rest
df = df.loc[df['Delete'] != True]



################ REMOVE OUTLIERS WITH DEVIAITON >=3 #########################

def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

for i in df['Store'].unique():
    df.loc[(df['Set'] == 1) & (df['Store'] == i) & (df['Open'] == 1), 'Outlier'] = \
        mad_based_outlier(df.loc[(df['Set'] == 1) & (df['Store'] == i) & (df['Open'] == 1)]['Sales'], 3)

# no_stores_to_check = 10

# plt.rcParams["figure.figsize"] = [20,no_stores_to_check*5]

# for i in range(1,no_stores_to_check+1):
#     stor = i

#     # Normal sales
#     X1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1) & (df['Outlier'] == False)]
#     y1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1) & (df['Outlier'] == False)]['Sales']

#     # Outliers
#     X2 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1) & (df['Outlier'] == True)]
#     y2 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1) & (df['Outlier'] == True)]['Sales']

#     Xt = df.loc[(df['Store'] == stor)]
    
#     plt.subplot(10,5,i)
#     plt.plot(X1['Date'], y1, '-')
#     plt.plot(X2['Date'], y2, 'r.')
#     plt.title(i)
#     plt.axis('off')


############################ SECOND SEGMENT START ###############################

############ Split Data into Training and Test for Filling in the Outliers ######################
X_train, X_test, y_train, y_test = train_test_split(df.loc[(df['Set'] == 1) & (df['Open'] == 1) & (df['Outlier'] == False)][features_x],
                                                    df.loc[(df['Set'] == 1) & (df['Open'] == 1) & (df['Outlier'] == False)][features_y],
                                                    test_size=0.1, random_state=seed) #test size is on 10% of data set

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

num_round = 250 #20000 #number of rounds for the algorithm to take
evallist = [(dtrain, 'train'), (dtest, 'test')]

param = {'bst:max_depth':12, #max depth of a tree. Controls overfitting as higher depth will allow model to learn relations very specific to a particular sample.
         'bst:eta':0.01, #learning rate. Makes model more robust by shrinking weights on each step
         'subsample':0.8, #denotes the fraction of observations to be randomly sampled for each tree. Lower values make algo more conservative, prevents overfitting but too small might lead to overfitting
         'colsample_bytree':0.7, #denotes fraction of columns to be randomly sampled for each tree
         'silent':1, #silent mode acticate if 1. 0 activates messages, which helps to understand the model
         'objective':'reg:linear', #defines loss function to be minimized. Could be binary:logstic (returns predicted probability, not class), multi:softprob (returns predicted probability of each data point belonging to each class), multi:softmax (returns predicted class, not probabilities). Add num_class parameter.
         # 'nthread':-1, #manually entereed, or leave this blank to run on all cores available.
         'seed':seed} #random seed number, can be used for generating reproducible results and for parameter tuning


         #gamma - split only when resulting split gives positive reduction in loss function. Makes algorithm conservative. Values can vary depending on the loss function and should be tuned.
         #lambda - L2 regularization term on weights, should be explored to reduce overfitting
         #alpha - L1 regularization (lasso), can be used in the case of very high dimensionality so that algo runs faster when implemented


         #eval metric - rmse for regression, error for classification. Has mae (mean absolute error), logloss (negative log-likelihood), error (binary classification), merror (multiclass classification error rate), mlogloss(multiclass logloss), auc (area under curve)



plst = param.items()

bst = xgb.train(plst, dtrain, num_round, evallist, feval=rmspe_xg, verbose_eval=250, early_stopping_rounds=250)

dpred = xgb.DMatrix(df.loc[(df['Set'] == 1) & (df['Open'] == 1) & (df['Outlier'] == True)][features_x]) #takes Open and Outliers as main 

ypred_bst = bst.predict(dpred)

df.loc[(df['Set'] == 1) & (df['Open'] == 1) & (df['Outlier'] == True), 'SalesLog'] = ypred_bst #does this make a difference on next segment? Is this because the dataframe now has outliers and saleslog with the new predictions?
df.loc[(df['Set'] == 1) & (df['Open'] == 1) & (df['Outlier'] == True), 'Sales'] = np.exp(ypred_bst) - 1




############################ SECOND SEGMENT END ###############################


### You see the result being lower than before, but most of them are still pretty high
# no_stores_to_check = 10

# plt.rcParams["figure.figsize"] = [20,no_stores_to_check*5]

# for i in range(1,no_stores_to_check+1):
#     stor = i

#     # Normal sales
#     X1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1) & (df['Outlier'] == False)]
#     y1 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1) & (df['Outlier'] == False)]['Sales']

#     # Outliers
#     X2 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1) & (df['Outlier'] == True)]
#     y2 = df.loc[(df['Set'] == 1) & (df['Store'] == stor) & (df['Open'] == 1) & (df['Outlier'] == True)]['Sales']

#     Xt = df.loc[(df['Store'] == stor)]
    
#     plt.subplot(10,5,i)
#     plt.plot(X1['Date'], y1, '-')
#     plt.plot(X2['Date'], y2, 'r.')
#     plt.title(i)
#     plt.axis('off')


###################### Generating some extra store data from sales ########################
### Get total sales, customers and open days per store
store_data_sales = df.groupby([df['Store']])['Sales'].sum()
store_data_customers = df.groupby([df['Store']])['Customers'].sum()
store_data_open = df.groupby([df['Store']])['Open'].count()

### Calculate sales per day, customers per day and sales per customers per day
store_data_sales_per_day = store_data_sales / store_data_open
store_data_customers_per_day = store_data_customers / store_data_open
store_data_sales_per_customer_per_day = store_data_sales_per_day / store_data_customers_per_day

df_store = pd.merge(df_store, store_data_sales_per_day.reset_index(name='SalesPerDay'), how='left', on=['Store'])
df_store = pd.merge(df_store, store_data_customers_per_day.reset_index(name='CustomersPerDay'), how='left', on=['Store'])
df_store = pd.merge(df_store, store_data_sales_per_customer_per_day.reset_index(name='SalesPerCustomersPerDay'), how='left', on=['Store'])

store_features = ['Store', 'SalesPerCustomersPerDay']
#store_features = ['Store', 'SalesPerDay', 'CustomersPerDay', 'SalesPerCustomersPerDay']

features_x = list(set(features_x + store_features))

print features_x

df = pd.merge(df, df_store[store_features], how='left', on=['Store'])



##################### BEGIN RANDOM SAMPLING ##############################
X_train, X_test, y_train, y_test = train_test_split(df.loc[(df['Set'] == 1) & (df['Open'] == 1)][features_x],
                                                    df.loc[(df['Set'] == 1) & (df['Open'] == 1)][features_y],
                                                    test_size=0.1, random_state=seed)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

num_round = 1250 #20000
evallist = [(dtrain, 'train'), (dtest, 'test')]

param = {'bst:max_depth':12,
         'bst:eta':0.0095,
         'subsample':0.8,
         'colsample_bytree':0.7,
         'silent':1, 
         'objective':'reg:linear',
         # 'nthread':6,
         'seed':seed}

plst = param.items()

print datetime.datetime.now().time()

#algo runs twice coz of removal of outliers and feature engineering. Comment out and see difference.
bst1 = xgb.train(plst, dtrain, num_round, evallist, feval=rmspe_xg, verbose_eval=250, early_stopping_rounds=250)

# xgb.plot_importance(bst1)

################ EXPORTING DATA ###############
X_submit = df.loc[df['Set'] == 0]

dsubmit = xgb.DMatrix(X_submit[features_x])

ypred_bst = bst1.predict(dsubmit)

df_ypred = X_submit['Id'].reset_index()

del df_ypred['index']

df_ypred['Id'] = df_ypred['Id'].astype('int')

df_ypred['Sales'] = (np.exp(ypred_bst) - 1) * 0.985 #apply constant factor to improve score. Can use 0.995 too.

df_ypred.sort_values('Id', inplace=True)
df_ypred[['Id', 'Sales']].to_csv('Rossmann_' + str(num_round) + ' rounds.csv', index=False)

print "done!"
print datetime.datetime.now().time()


