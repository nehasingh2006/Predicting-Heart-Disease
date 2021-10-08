import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")
data=pd.read_csv('heart.csv')
data = np.array(data)
X=data[0:,0:-1]
y=data[0:,-1]
print(X)
print(y)
y = y.astype('int')
X = X.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
heart=KNeighborsClassifier(n_neighbors=1)
heart.fit(X_train,y_train)
pickle.dump(heart,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
