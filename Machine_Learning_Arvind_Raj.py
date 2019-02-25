import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import confusion_matrix

filename = 'data_ml.csv'
data = pd.read_csv(filename)
pd.set_option('display.max_columns', 10)


# Data analysis to gather info on different variables
print(data.head())
print(data.describe())
print(data.info())

# Analyze the different duration of jobs
data['duration'].value_counts().plot(kind='bar')
plt.title("duration of jobs")
plt.xlabel("duration")
plt.ylabel("count")
sns.despine

# Drop outlier rows from 'models' & 'params' columns 
# These indexes are found using the .idxmax() method for the above columns
data_new = data.drop(data.index[[1053, 1299,2005, 2843, 2160]])

# Fill NaN values with ffill and median values
data_type_nan = data_new[data_new['type'].isnull()]
data_type_no_nan = data_new.type.fillna(method='ffill')
data_new['type'] = data_type_no_nan

data_params_nan = data_new[data_new['params'].isnull()]
data_params_no_nan = data_new.params.fillna(data_new.params.median())
data_new['params'] = data_params_no_nan

#We have full dataset without any null values or outliers
#Define the different models to see which will fit best :

linreg = LinearRegression() 
logreg = LogisticRegression()
gbr = ensemble.GradientBoostingRegressor(n_estimators = 50, max_depth = 8, min_samples_split = 2, learning_rate = 0.1, loss = 'ls')
tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

labels = data_new['duration']
version_1 = data_new.drop(['duration','Unnamed: 0'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(version_1, labels, test_size = 0.10, random_state = 2)

#Linear Regression
linreg.fit(x_train, y_train)
lin_reg_score = linreg.score(x_test, y_test)
lin_reg_predict = linreg.predict(x_test)
print("LinearRegression Score: " + str(lin_reg_score))

#Logistic regression
logreg.fit(x_train, y_train)
log_reg_score = logreg.score(x_test, y_test)
log_reg_predict = logreg.predict(x_test)
log_true = np.array(data_new.duration)
log_matrix = confusion_matrix(y_test, log_reg_predict)
print("Logistic Matrix: \n" + str(log_matrix))
print("LogisticRegression Score: " + str(log_reg_score))

#Gradient Boosted regression
gbr.fit(x_train, y_train)
score_boost = gbr.score(x_test,y_test)
boost_predict = gbr.predict(x_test)
print("Boosted Score: " + str(score_boost))

#Decision tree regression
tree_clf.fit(x_train, y_train)
tree_score = tree_clf.score(x_test, y_test)
tree_predict = tree_clf.predict(x_test)
tree_matrix = confusion_matrix(y_test, tree_predict)
print("Tree Matrix: \n" + str(tree_matrix))
print("Tree Score: "+ str(tree_score))
