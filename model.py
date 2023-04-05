import pandas as pd 

#importing the data
data = pd.read_csv('Iris.csv', index_col=0)

#renaming the columns
data = data.rename(columns={"SepalLengthCm": "sepal_length", "SepalWidthCm": "sepal_width", "PetalLengthCm": "petal_length", "PetalWidthCm": "petal_width", "Species": "species"})

print(data.head())

print(data.info())

print(data.describe())

rows, col = data.shape
print("Rows : %s, column : %s" % (rows, col))

print(data['species'].value_counts())


#make a copy of the species column
data['species_detailed'] = data['species']

#replacing the two not setosa species with "not Setosa"
dic_setosa = {'Iris-versicolor': 'not Setosa', 'Iris-virginica': 'not Setosa', 'Iris-setosa': 'Setosa'}
data = data.replace({"species": dic_setosa})

print(data['species'].value_counts())

print(data.columns)


#importing a tool to split the dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


features = data.drop(columns=['species', 'species_detailed'])
labels = data['species']

#use the label binarizer from sklearn
#before ['Setosa', 'Not Setosa', 'Not Setosa']
#after [1,0,0]
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

#split the features and labels in to a Train (80%) and a Test (20%) set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

#split the train set into a Train (75%) and a Validation (25%) set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

#in the end we have: Train (60%), Validation (20%), Test (20%)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test,y_test_pred)

print('Accuracy: {} '.format(accuracy))


# save the model to disk
import pickle
filename = 'iris_model.sav'
pickle.dump(model, open(filename, 'wb'))
