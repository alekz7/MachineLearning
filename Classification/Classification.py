# This is a test of Classification Machine Learning
# Alekz7 based on Siraj Raval
# the question is, the new element is Male or Female?
from sklearn import tree

X = [[181,80,44],[177,70,43],[160,60,38],[154,54,37]]
Y = ['male','female','female','female']
print('X size: ' + str(len(X)) + ', Y size: ' + str(len(Y)))

#Definicion
clf = tree.DecisionTreeClassifier()

#Training
clf = clf.fit(X,Y)

prediction = clf.predict([[194,45,36]])

print(prediction)