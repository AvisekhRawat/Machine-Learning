import numpy as np
import pandas as pd	
pd.options.mode.chained_assignment = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

rf=joblib.load('rf1')
data=pd.read_csv('titanic_test.csv')
new_data=data[['pclass','sex']]
print(new_data.head())

new_data['pclass'].replace('1st',1,inplace=True)
new_data['pclass'].replace('2nd',2,inplace=True)
new_data['pclass'].replace('3rd',3,inplace=True)
new_data['sex']=np.where(new_data['sex']=='male',1,0)
print(new_data.head())
## pre
pre=rf.predict(new_data)

# comparing with external result file
tr=np.loadtxt('titanic_results.txt',dtype='int32')
e=np.equal(tr,pre)
s=np.sum(e) 
c=len(e)

print('Accuracy ',s/c*100)
