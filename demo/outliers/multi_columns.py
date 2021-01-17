import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Removing multi dimensional outliers
def train_isolation_forest(df, features, n_estimators=1000, bootstrap=False, max_samples='auto'):
    '''This function takes as input the DataFrame and corresponding features and fit Isolation forest'''

    X_train = df[features]

    clf=IsolationForest(n_estimators=n_estimators, max_samples=max_samples, \
       bootstrap=bootstrap, n_jobs=-1, random_state=42, verbose=0)

    clf.fit(X_train)

    return clf

def get_anomaly_and_score(df, features, clf):
    '''This function takes as input DataFrame, trained model for outliers and imputer used in training
    and update columns with anomaly (1 being normal and -1 being outlier)'''

    X_train = df[features]

    pred = clf.predict(X_train)

    df['anomaly']=pred

    pred_score = clf.score_samples(X_train)

    df['score_anomaly']=pred_score

    return df

def get_outliers_index(df,mode = 'normal', threshold = -0.5 , percent = 0.5):
    '''This function takes as input a DataFrame and return indexes of outliers and not outliers.
    Three modes: normal mode that output all outliers, threshold mode that ouput based on a threshold
    and percent that output based on a percent of outliers'''

    if mode == 'normal':

        outliers=df.loc[df['anomaly']==-1]
        outlier_index=list(outliers.index)
        clean=df.loc[df['anomaly']==1]
        clean_index=list(clean.index)

    elif mode == 'threshold':

        outliers=df.loc[df['score_anomaly']<threshold]
        outlier_index=list(outliers.index)
        clean=df.loc[df['score_anomaly']>=threshold]
        clean_index=list(clean.index)

    elif mode == 'percent':

        threshold = df.sort_values(by='score_anomaly')[:int(df.shape[0]*(percent/100))]['score_anomaly'].values[-1]
        outliers=df.loc[df['score_anomaly']<threshold]
        outlier_index=list(outliers.index)
        clean=df.loc[df['score_anomaly']>=threshold]
        clean_index=list(clean.index)

    return outlier_index, clean_index


def plot_anomaly_pca(df, features, outlier_index, clean_index, mode = '3D'):
    '''This function takes dataFrame as input, with features of interest and indexes of outliers
    and plot the 3D or 2D (depending on the mode) PCA plot of outlier vs normal
    '''

    if mode == '3D':
        pca = PCA(n_components=3)  # Reduce to k=3 dimensions
        scaler = StandardScaler()
        #normalize the features
        X_train = df[features]

        X = scaler.fit_transform(X_train)

        #Reduce to 3 dimension with PCA
        X_reduce = pca.fit_transform(X)

        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel("x_composite_3")

        # Plot the compressed data points
        ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=4, lw=1, label="inliers",c="green")

        # Plot x's for the ground truth outliers
        ax.scatter(X_reduce[outlier_index,0],X_reduce[outlier_index,1], X_reduce[outlier_index,2],
                   lw=2, s=60, marker="x", c="red", label="outliers")
        ax.legend()
        plt.show()

    elif mode == '2D':

        plt.figure(figsize=(10,8))
        pca = PCA(n_components=3)  # Reduce to k=3 dimensions
        scaler = StandardScaler()
        #normalize the features
        X_train = df[features]
        X = scaler.fit_transform(X_train)
        #Reduce dimension with PCA
        X_reduce = pca.fit_transform(X)

        # Plot the compressed data points
        plt.scatter(X_reduce[clean_index,0],X_reduce[clean_index,1],label='normal points')
        plt.scatter(X_reduce[outlier_index,0],X_reduce[outlier_index,1],c='red', label='predicted outliers')
        plt.legend(loc="upper right")
        plt.show()

    return
