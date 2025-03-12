import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import joblib


def preprocess(user,movies,y_train):
    try:
        #unscaled_user=user
        #unscaled_movie=movies
        scalerUser=StandardScaler()
        scalerUser.fit(user)
        user=scalerUser.transform(user)
        joblib.dump(scalerUser, './model/scalerUser.pkl')
        scalerItem=StandardScaler()
        scalerItem.fit(movies)
        movies=scalerItem.transform(movies)
        joblib.dump(scalerItem, './model/scalerItem.pkl')
        scalerTarget = MinMaxScaler((-1, 1))
        y_train_array = y_train.to_numpy().reshape(-1, 1)
        scalerTarget.fit(y_train_array)
        y_train = scalerTarget.transform(y_train_array)
    
        user_train,user_test=train_test_split(user,test_size=0.2,random_state=1)
        movie_train,movie_test=train_test_split(movies,test_size=0.2,random_state=1)
        y_train,y_test=train_test_split(y_train,test_size=0.2,random_state=1)
      
        return user_train,user_test,movie_train,movie_test,y_train,y_test,scalerUser,scalerItem,scalerTarget
    except Exception as e:
        error_message = f"Failed to preprocess model: {e}"
        print(error_message)
