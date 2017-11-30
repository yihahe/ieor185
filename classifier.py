# @yihahe 11/27
# run preprocessing.py first if data (mock_data.csv) has changed

import pandas as pd
from sklearn import svm
from sklearn import neural_network as nn

# split test_df and train_df into x and y
test_df = pd.read_csv('csv/test.csv')
train_df = pd.read_csv('csv/train.csv')
test_df_y = test_df['Rich?']
train_df_y = train_df['Rich?']
test_df_x = test_df.drop('Rich?', 1)
train_df_x = train_df.drop('Rich?', 1)

# support vector machine classifier
clf = svm.SVC()
clf.fit(train_df_x, train_df_y)
print (clf.predict(test_df_x))

# linear SVC
lin_clf = svm.LinearSVC()
lin_clf.fit(train_df_x, train_df_y)
print (lin_clf.predict(test_df_x))

# MLP (multi-layer perception) Neural Network Classifier
mlp_clf = nn.MLPClassifier()
mlp_clf.fit(train_df_x, train_df_y)
print (mlp_clf.predict(test_df_x))


print list(test_df_y.values)
