import pandas as pd
from sklearn import svm
from sklearn import neural_network as nn
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder
import pickle

class Train_API:
    def __init__(self):
        self.csv = pd.DataFrame()

    def save(self, csv):
        self.csv = csv

    def everything(self):
        le_dict = {}

        # transform categorical columns
        cat_col = ['Name', 'FB Current City', 'LinkedIn 500+?', 'School']
        for column_name in cat_col:
            le = LabelEncoder()
            self.csv[column_name] = le.fit_transform(self.csv[column_name])
            le_dict[column_name] = le

        # save label_encoder dictionary in pickle file
        with open('le_dict.pickle', 'wb') as handle:
            pickle.dump(le_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # split data into 60/40 train-test
        train_df, test_df = tts(self.csv, train_size=.6, test_size=.4)

        # save training and test data
        pd.DataFrame.to_csv(train_df, 'csv/train.csv')
        pd.DataFrame.to_csv(test_df, 'csv/test.csv')
        pd.DataFrame.to_csv(self.csv, 'csv/transform.csv')
        return self.classifier()

    def classifier(self):
        # Section Classifier.2 - preprocessing
        # load label encoders
        with open('le_dict.pickle', 'rb') as f:
            le_dict = pickle.load(f)

        # return label encoder for names
        le = le_dict['Name']

        # split test_df and train_df into x and y
        test_df = pd.read_csv('csv/test.csv')
        train_df = pd.read_csv('csv/train.csv')
        transform_df = pd.read_csv('csv/transform.csv')
        test_df_y = test_df['Rich?']
        train_df_y = train_df['Rich?']
        transform_df_y = transform_df['Rich?']
        test_df_x = test_df.drop('Rich?', 1)
        train_df_x = train_df.drop('Rich?', 1)
        transform_df_x = transform_df.drop('Rich?', 1)

        # inverse transform name column for test and whole
        reverse_test_names = le.inverse_transform(test_df_x['Name'])
        reverse_test_names_whole = le.inverse_transform(transform_df_x['Name'])

        # support vector machine classifier
        clf = svm.SVC()
        clf.fit(train_df_x, train_df_y)
        clf_result = clf.predict(test_df_x)
        clf_result_whole = clf.predict(transform_df_x)
        print ('clf: ', clf.score(transform_df_x, transform_df_y))

        # linear SVC
        lin_clf = svm.LinearSVC()
        lin_clf.fit(train_df_x, train_df_y)
        lin_clf_result = lin_clf.predict(test_df_x)
        lin_clf_result_whole = lin_clf.predict(transform_df_x)
        print ('lin_clf: ', lin_clf.score(transform_df_x, transform_df_y))

        # MLP (multi-layer perception) Neural Network Classifier
        mlp_clf = nn.MLPClassifier()
        mlp_clf.fit(train_df_x, train_df_y)
        mlp_clf_result = mlp_clf.predict(test_df_x)
        mlp_clf_result_whole = mlp_clf.predict(transform_df_x)
        print ('mlp_clf: ', mlp_clf.score(transform_df_x, transform_df_y))

        # clean up result and compare predictions with actual for test
        result = np.vstack((reverse_test_names, clf_result, lin_clf_result, mlp_clf_result, test_df_y)).T
        df_result = pd.DataFrame(result, columns=['Name', 'SVM Pred', 'Linear Pred', 'MLP Pred', 'Actual'])
        df_result = df_result.set_index(df_result['Name'])
        df_result.drop('Name', axis=1, inplace=True)
        print (df_result)

        # clean up result and compare predictions with actual for all of mock data
        reverse_test_names = le.inverse_transform(transform_df_x['Name'])
        result = np.vstack((reverse_test_names_whole, clf_result_whole, lin_clf_result_whole,
                            mlp_clf_result_whole, transform_df_y)).T
        df_result = pd.DataFrame(result, columns=['Name', 'SVM Pred', 'Linear Pred', 'MLP Pred', 'Actual'])
        df_result = df_result.set_index(df_result['Name'])
        df_result.drop('Name', axis=1, inplace=True)
        print (df_result)
        return df_result
