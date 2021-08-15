import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold

COMPETITION_NAME = "siimcovid19-512-img-png-600-study-png"
load_dir = f"../input/{COMPETITION_NAME}/"
df = pd.read_csv('../input/train_v1.csv')#'../input/siim-covid19-detection/train_study_level.csv')
df.head()

df.rename(columns={'Negative for Pneumonia':'0','Typical Appearance':'1',"Indeterminate Appearance":'2',
                   "Atypical Appearance":"3"}, inplace=True)

labels = []
def get_label(row):
    for c in df.columns:
        if row[c]==1:
            labels.append(int(c))
df.apply(get_label, axis=1)
print("label modified")


labels = {'label':labels}
study_label = pd.DataFrame(labels)
train_study = pd.concat([df, study_label], axis = 1)
print(train_study)

del train_study ['0'];del train_study ['1'];del train_study ['2'];del train_study ['3']
train_study