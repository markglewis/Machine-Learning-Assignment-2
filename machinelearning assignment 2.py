#machine learning steps copy paste into ipython

from __future__ import print_function, division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
palette = sns.color_palette('deep', 5)
palette[1], palette[2] = palette[2], palette[1]
%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')

action_type = pd.get_dummies(train['ACTION_TYPE'])
htm = pd.get_dummies(train['HTM'])
shot_type = pd.get_dummies(train['SHOT_TYPE'])
shot_zone_area = pd.get_dummies(train['SHOT_ZONE_AREA'])
shot_zone_basic = pd.get_dummies(train['SHOT_ZONE_BASIC'])
shot_zone_range = pd.get_dummies(train['SHOT_ZONE_RANGE'])

train = train.drop(['GAME_ID', 'PLAYER_ID', 'TEAM_ID','ACTION_TYPE', 'EVENT_TYPE','GAME_DATE','HTM', 'GAME_EVENT_ID','SHOT_TYPE','SHOT_ZONE_AREA',
                        'SHOT_ZONE_BASIC', 'SHOT_ZONE_RANGE','TEAM_NAME','VTM','PLAYER_NAME','SHOT_ATTEMPTED_FLAG'], axis=1)
											
train = pd.concat([train, htm, shot_type, shot_zone_area,shot_zone_basic, shot_zone_range], axis=1)
train.head()

action_type = pd.get_dummies(val['ACTION_TYPE'])
htm = pd.get_dummies(val['HTM'])
shot_type = pd.get_dummies(val['SHOT_TYPE'])
shot_zone_area = pd.get_dummies(val['SHOT_ZONE_AREA'])
shot_zone_basic = pd.get_dummies(val['SHOT_ZONE_BASIC'])
shot_zone_range = pd.get_dummies(val['SHOT_ZONE_RANGE'])

val = val.drop(['GAME_ID', 'PLAYER_ID', 'TEAM_ID','ACTION_TYPE', 'EVENT_TYPE','GAME_DATE','HTM', 'GAME_EVENT_ID','SHOT_TYPE','SHOT_ZONE_AREA',
                        'SHOT_ZONE_BASIC', 'SHOT_ZONE_RANGE','TEAM_NAME','VTM','PLAYER_NAME','SHOT_ATTEMPTED_FLAG'], axis=1)
											
val = pd.concat([val, htm, shot_type, shot_zone_area,shot_zone_basic, shot_zone_range], axis=1)
train.head()


X_train = train.drop('SHOT_MADE_FLAG', axis=1)
y_train = train['SHOT_MADE_FLAG']	


												
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


X_test = val.drop('SHOT_MADE_FLAG', axis=1)
y_test = val['SHOT_MADE_FLAG']
predictions = logmodel.predict(X_test)

solution = pd.read_csv('solution_no_answer.csv')
output = solution['GAME_EVENT_ID']

action_type = pd.get_dummies(solution['ACTION_TYPE'])
htm = pd.get_dummies(solution['HTM'])
shot_type = pd.get_dummies(solution['SHOT_TYPE'])
shot_zone_area = pd.get_dummies(solution['SHOT_ZONE_AREA'])
shot_zone_basic = pd.get_dummies(solution['SHOT_ZONE_BASIC'])
shot_zone_range = pd.get_dummies(solution['SHOT_ZONE_RANGE'])

solution = solution.drop(['GAME_ID', 'PLAYER_ID', 'TEAM_ID','ACTION_TYPE', 'EVENT_TYPE','GAME_DATE','HTM', 'GAME_EVENT_ID','SHOT_TYPE','SHOT_ZONE_AREA',
                        'SHOT_ZONE_BASIC', 'SHOT_ZONE_RANGE','TEAM_NAME','VTM','PLAYER_NAME','SHOT_ATTEMPTED_FLAG'], axis=1)
											
solution = pd.concat([solution, htm, shot_type, shot_zone_area,shot_zone_basic, shot_zone_range], axis=1)
solution.head()

X_Solution = solution
predictions = logmodel.predict(X_Solution)




dataframe=pd.DataFrame(predictions, columns=['SHOT_MADE_FLAG']) 
output = pd.concat([output, dataframe], axis=1)
output.to_csv('solution.csv' ,  index = False)