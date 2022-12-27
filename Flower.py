
#importing the libraries
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#reading the data
data = pd.read_csv("C:/Datasets/Iris.csv")

#defining the feature columns
#SepalLength, SepalWidth, PetalLength, PetalWidth
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

#defining the target column
target_cols = 'Species'

#extracting features and target from the data
X = data[feature_cols] # Features
y = data[target_cols] # Target

#split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1,shuffle=True)

#create the model using the decision tree classifier
model = DecisionTreeClassifier()

#train the model using
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#calculating accuracy
accuracy = accuracy_score(y_test, y_pred)


#plotting graph
plt.scatter(y_test, y_pred)
plt.title('Flower Decision Tree Classifier')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

#printing accuracy
print('Accuracy:', accuracy)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
