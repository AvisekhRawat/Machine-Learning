import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

pd.options.mode.chained_assignment = None
data=pd.read_csv('titanic_train.csv')
print(data.shape)
#print (data.head(10))
#print(data.columns)
new_data=data[["pclass","sex"]]
print(new_data.head())
new_output=data[["survived"]]
print(new_output.head())
new_data['pclass'].replace('3rd',3,inplace=True)
new_data['pclass'].replace('2nd',2,inplace=True)
new_data['pclass'].replace('1st',1,inplace=True)
#new_data['sex'].replace('female',0,inplace=True)
#new_data['sex'].replace('male',1,inplace=True)
new_data['sex']= np.where(new_data['sex']=='female',0,1)
X_tr, X_te ,y_tr, y_te= train_test_split(new_data,new_output,test_size=0.33,random_state=42)

print(new_data.head())

print('After train_test_split')
print(X_tr.shape)
print(X_te.shape)
print(y_tr.shape)
print(y_te.shape)

rf=RandomForestClassifier(n_estimators=100)
rf.fit(X_tr,y_tr)
acc=rf.score(X_te,y_te)
print(acc*100)

joblib.dump(rf,'rf1',compress=9)
