import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
pio.templates.default = "plotly_white"

data = pd.read_csv("/home/ambar/Documents/SEM6/ML/ad_10000records.csv")
print(data.head())

data["Clicked on Ad"] = data["Clicked on Ad"].map({0: "No", 1: "Yes"})

fig = px.box(data, 
             x="Daily Time Spent on Site",  
             color="Clicked on Ad", 
             title="Click Through Rate based Time Spent on Site", 
             color_discrete_map={'Yes':'blue', 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

data["Clicked on Ad"].value_counts()
click_through_rate = 4917 / 10000 * 100
print(click_through_rate)

data["Gender"] = data["Gender"].map({"Male": 1, 
                               "Female": 0})

x=data.iloc[:,0:7]
x=x.drop(['Ad Topic Line','City'],axis=1)
y=data.iloc[:,9]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,
                                           test_size=0.2,
                                           random_state=4)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain.values, ytrain.values)

y_pred = model.predict(xtest)

from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(ytest,y_pred))

# cf_matrix = confusion_matrix(ytest, y_pred)
# # print("Confusion matrix:")
# # print(cf_matrix)
# print(classification_report(ytest, y_pred))
# cf_matrix = metrics.confusion_matrix(ytest,y_pred)

# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = [False, True])

# cm_display.plot()
# plt.show()

print("Ads Click Through Rate Prediction : ")
a = float(input("Daily Time Spent on Site: "))
b = float(input("Age: "))
c = float(input("Area Income: "))
d = float(input("Daily Internet Usage: "))
e = input("Gender (Male = 1, Female = 0) : ")

features = np.array([[a, b, c, d, e]])
print("Will the user click on ad = ", model.predict(features))
