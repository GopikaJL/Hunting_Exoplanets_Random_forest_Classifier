#Importing the modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

train_df = pd.read_csv('/exoTrain.csv')
train_df.head()
test_df = pd.read_csv('/exoTest.csv')
test_df.head()

#Check For The Missing Values
train_df.isnull().sum()

#Second method
num_missing_values = 0
for i in train_df.columns:
  for j in train_df[i].isnull():
    if j == True:
      num_missing_values += 1
num_missing_values

#Non missing values
non_missing_values = 0
for i in train_df.columns:
  for j in rain_df[i].isnull():
    if j == True:
      non_missing_values += 1
non_missing_values

#Detecting the curve according to transit principle
star_0 = train_df.iloc[0,:]
plt.figure(figsize=(16, 4))
x = np.arange(1, 3198)
#3198 is the total number of rows of the dataframe
y = star_0[1:]
plt.scatter(x, y)
plt.show() 

#Seperating the feature and target variable for training data
x_train = train_df.iloc[:, 1:]
x_train.head()
y_train = train_df.iloc[:, 0] 
y_train.head()

#Seperating the feature and target variable for testing data
x_test = test_df.iloc[:, 1:]
x_test.head()
y_test = test_df.iloc[:, 0] 
y_test.head()

#Training the model
r = RandomForestClassifier(n_jobs=-1, n_estimators=50)
r.fit(x_train, y_train)
r.score(x_train, y_train)

#Evaluating the model performance 
y_pred = r.predict(x_test)
y_pred = pd.Series(y_pred)

#Validation using classification report and confucion matrix
print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)

#Since the validation is showing no classification result for class 2 we can check whether dataset is balanced or not
ytest.value_counts()
ytrain.value_counts()

#performing normalisation on the dataset
def normalise(x):
  norm = (x - x.mean()) / (x.max() - x.min())
  return norm
  
#Applyng nirmalisation on the training and testing data
nortrain_df = train_df.iloc[:, 1:].apply(normalise, axis=1)
#Inserting the LABEL column back to dataframe since the LABEL column already have the normalized data of 0 and 1 we have removed it at the stage of normalization
nortrain_df.insert(loc=0, column='LABEL', value=train_df['LABEL'])
nortrain_df.head()
nortest_df = test_df.iloc[:, 1:].apply(normalise, axis=1)
nortest_df.insert(loc=0, column='LABEL', value=test_df['LABEL'])
nortest_df.head()

#Creating a function for fast fourier transformation
def fft(s):
  fftstar = np.fft.fft(s, n=len(s))
  return np.abs(fftstar)

#Applying Fast fourier transformation to make the data convert from time to frequency domain
fft_train = nortrain_df.iloc[:, 1:].T.apply(fft, axis=0).T.head()
fft_test= nortest_df.iloc[:, 1:].T.apply(fft, axis=0).T.head()

#Apply sampling method to develop a balanced dataset for both classes
sm = SMOTE(sampling_strategy=1)
xsm,ysm = sm.fit_resample(fft_train, y_train)

#Check the balanced data
ysm.value_counts()

#Applying Random Forest classifier on the processed dataset
r = RandomForestClassifier(n_jobs=-1, n_estimators=50)
r.fit(xsm,ysm)
print(r.score(xsm, ysm))
y_pred = rf_clf.predict(fft_test)
y_pred

confusion_matrix(y_test, y_pred)
print( classification_report(y_test, y_pred)

#Evaluation using XGBoost classifier
import xgboost as xg
x = xg.XGBClassifier()
x.fit(xsm, ysm)
ypred1 = x.predict(fft_test)
ypred1

confusion_matrix(y_test, ypred1)
print( classification_report(y_test, ypred1)
So, XGBoost classifier has much higher recall and f1-scores compared to Random Forest Classifier in this dataset. So we conclude XGBoost is the best machine learning model suited for this exoplanet dataset.
