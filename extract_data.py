import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('features-clear-missing-value.csv')
test = pd.read_csv('test_data.csv')

print(test.head())

# tach nhan(y) va du lieu(x) training tu data
X = data.iloc[:, 6:] 
y = data['category'] 

# tach nhan(y) va du lieu(x) testing tu test
#knn 
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
y_pred = knn.predict(X)
acc = accuracy_score(y, y_pred)

print('Accuracy: ', acc)

# save model to file
joblib.dump(knn, 'knn.pkl')
# save result to a column name 'acc' in test_data.csv

test['acc'] = acc
test.to_csv('test_data.csv', index=False)

