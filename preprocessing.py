# @yihahe 11/27

import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder
import pickle

# read in csv, turn categorical / qualitative data to quantitative data using Label Encoder
pre_df = pd.read_csv('csv/mock_data.csv')
le_dict = {}
for column_name in pre_df.columns.values:
    le = LabelEncoder()
    pre_df[column_name] = le.fit_transform(pre_df[column_name])
    le_dict[column_name] = le

# save label_encoder dictionary in pickle file
with open('le_dict.pickle', 'wb') as handle:
    pickle.dump(le_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# split data into 60/40 train-test
train_df, test_df = tts(pre_df, train_size=.6, test_size=.4)

# save training and test data
train_csv = pd.DataFrame.to_csv(train_df, 'csv/train.csv')
test_csv = pd.DataFrame.to_csv(test_df, 'csv/test.csv')
