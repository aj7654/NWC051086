import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib as plt

# from pandas import ExcelWriter
# from pandas import ExcelFile
# import pandas_profiling 

# from sklearn.feature_selection import RFE
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import ElasticNet
# from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection import KFold

from joblib import (dump, load)
import pickle

def interface():
    df=pd.read_csv('test_data.csv')
    print(df)

    x_test = df.drop(['sl.no','danger'],axis=1)
    y_test = df['danger']
    print(x_test)
    print(y_test)

    model = pickle.load(open('save.pkl', 'rb'))

    # model = load('model_joblib.joblib')
    print(model.score(x_test,y_test))

if __name__ == '__main__':
    interface()

