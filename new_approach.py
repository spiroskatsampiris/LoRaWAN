import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import math
import numpy as np
from scipy import optimize
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import normalize
from scipy import stats
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from plotly import express as px
import statistics
from mpl_toolkits.mplot3d import Axes3D
color = ["#ff0000", "#0073e6", "#ff751a", "#00ff00", "#e6e600", "#ff00ff","#0000e6",\
         "#006622", "#804000", "#6600ff", "#b30000", "#66ffff","#4d2600"]
def clean(data):
    data = data.drop(columns = ['codr']) #has only one value
    data = data.drop(columns = ['dev_eui']) #mapping one to one with dev_addr
    data['dev_nonce'] = data['dev_nonce'].fillna(0)
    data.loc[data['dev_nonce'] != 0,'dev_nonce'] = 1
    data = data.drop(columns = ['rssi']) #duplicated data from rssic
    data = data.drop(columns = ['created_at','tmms','value_minutes','tmst','time']) #duplicated timestamps
    data.ns_time = pd.to_datetime(data.ns_time)
    data.FCnt = data.FCnt.fillna(0)
    data['FCnt_diff'] = 0
    data['ns_time_diff'] = 0
    for dev in data.dev_addr.unique():
        if len(data[data.dev_addr == dev]) > 1:
            data.loc[data['dev_addr']==dev,'ns_time_diff']\
            = data[data['dev_addr']==dev]['ns_time'] - data[data['dev_addr']==dev]['ns_time'].shift(1)
            data.loc[data['dev_addr']==dev,'FCnt_diff']\
            = data.loc[data['dev_addr']==dev]['FCnt'] - data.loc[data['dev_addr']==dev]['FCnt'].shift(1) - 1
    data.FCnt_diff = data.FCnt_diff.fillna(0)
    data.loc[data['FCnt_diff']<0,'FCnt_diff'] = 0
    data.ns_time_diff = data.ns_time_diff.fillna(0)
    data = data.drop(columns=['FCnt','ns_time'])
    data = data.dropna()
    return data

def preprocess(data,id = False):

    categorical = ['datr','gateway','chan','ant','freq']
    numeric= ['dev_nonce','FCnt_diff','lsnr','rssic','rssis','rssisd','ns_time_diff']
    x_cat = data.loc[:,categorical]
    x_num = data.loc[:,numeric]

    #standardizing numeric data for scale
    scaler = StandardScaler()
    x_num = scaler.fit_transform(x_num)

    #label encoding for dev_addr
    enc_l = LabelEncoder()
    enc_l.fit(data.loc[:,'dev_addr'])
    dev_labels = enc_l.transform(data['dev_addr'])

    #one hot encoding for the rest

    enc_cat = OneHotEncoder()
    enc_cat.fit(x_cat.loc[:,x_cat.columns])

    #Transform
    onehotlabels = enc_cat.transform(x_cat.loc[:,x_cat.columns]).toarray()
    dev_labels = dev_labels.reshape((dev_labels.shape[0],1))
    if id == True:
        X = np.append(dev_labels, values=onehotlabels, axis=1)
        X = np.append(X,values = x_num,axis=1)
    else:
        X = np.append(onehotlabels, values=x_num, axis=1)
    return X


def findk(X):
    wcss = []
    dbs = []
    for i in range(2, 20):
        kmeans = KMeans(n_clusters=i, \
                        n_jobs=6, n_init=7, random_state=0).fit(X)
        # inertia_ : Sum of squared distances of samples to their closest cluster center.
        wcss.append(kmeans.inertia_)
        labels = kmeans.labels_
        dbs.append(davies_bouldin_score(X, labels))

        print('clustering with k = {} done!'.format(i))
    plt.plot(range(2, 20), wcss)
    plt.title('The Elbow Method Graph')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    # Plot the Davies-Bouldin graph
    plt.plot(range(2, 20), dbs)
    plt.title('The Davies-Bouldin Method Graph')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Davies-Bouldin Index')
    plt.show

def train_model(data,X,k):
    kmeans = KMeans(n_clusters=k, n_jobs=6, n_init=7, random_state=0)
    model = kmeans.fit(X)
    clusters = model.predict(X).reshape((len(X), 1))
    data['cluster'] = clusters
    data.cluster = data.cluster.astype(int)

    return data,clusters,model

def behavior(data):
    tt = data[['cluster','dev_addr']].groupby(['cluster','dev_addr']).size().unstack(fill_value=0)
    behavior = []
    for i in tt.columns:
        current_behavior = []
        current_percentile = []
        current_dev = tt.loc[:,i].values
        for j in current_dev:
            if j == 0:
                current_behavior.append(0)
            elif j <= np.percentile(current_dev, 50):
                current_behavior.append(-1)
            elif j < np.percentile(current_dev,90) and j >np.percentile(current_dev,50):
                current_behavior.append(1)
            elif j >= np.percentile(current_dev,90):
                current_behavior.append(2)
            current_percentile.append(j)
        current_percentile = np.array(current_percentile).reshape((len(current_percentile),1))
        current_behavior = np.array(current_behavior).reshape((len(current_behavior),1))

        behavior.append(np.append(current_behavior,current_percentile,axis=1))
    behavior = np.array(behavior)
    return(behavior)

def plot_behavior(data,dev):
    x = behavior(data)
    fig, ax = plt.subplots()
    barplot = plt.bar(x[dev,:,0],x[dev,:,1])
    ax.legend(barplot, ['-1 : anomalous behavior', '0: neutral behavior ', '1: normal behavior','2:typical behavior'])
    plt.show()

def cluster_counts_plot(data):
    tt = data[['cluster','dev_addr']].groupby(['cluster','dev_addr']).size().unstack(fill_value=0)
    tt_counted = tt.unstack().reset_index()
    tt_counted = tt_counted.rename(columns={0: "count"})
    sns.catplot(x='dev_addr', y='count',col = 'cluster', data=tt_counted,\
                kind='strip', height=4, legend=True, palette='hls',col_wrap=4)

def dev_per_cluster(data,dev):
    tt = data[['cluster','dev_addr']].groupby(['cluster','dev_addr']).size().unstack(fill_value=0)
    tt_counted = tt.unstack().reset_index()
    tt_counted = tt_counted.rename(columns={0: "count"})
    g = sns.FacetGrid(tt_counted.loc[tt_counted['dev_addr'] == dev] , col='cluster',col_wrap = 6,aspect = 1)
    g.map(sns.distplot, 'count', kde=False,hist=True,rug=True,bins=1);

def plot_hist(data,var):
    fig = plt.figure(figsize=(12, 4))
    sns.set(rc={'axes.facecolor': 'white', 'figure.facecolor': 'white', 'xtick.color': 'black', 'ytick.color': 'black' \
        , 'xtick.labelsize': '20', 'ytick.labelsize': '20'})

    sns.set_palette(sns.color_palette(color), 13)
    for i in data['cluster'].unique():
        sns.distplot(data[data['cluster'] == i][var], label='cluster', kde=False, bins=150, norm_hist=True)
    fig.legend(labels=np.sort(data['cluster'].unique()))

def plot_3d_animated(data,xx,yy,zz):
    fig = px.scatter_3d(data, x=xx, y=yy, z=zz, size='cluster', size_max=10,
                        opacity=1, color='cluster', width=1200, height=400)
    fig.show()
def plot_3d(data,x,y):

    sns.set_style("whitegrid", {'axes.grid': False})

    fig = plt.figure(figsize=(10, 10))

    # ax = Axes3D(fig) # Method 1
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[x], data[y], data['cluster'], c=data['cluster'], marker='o')
    ax.set_xlabel(x, labelpad=20)
    ax.set_ylabel(y, labelpad=20)
    ax.set_zlabel('Cluster', labelpad=20)
    ax.view_init(20, None)
    plt.show()

def lsnr_grid(data):
    g = sns.FacetGrid(data, col="cluster", col_wrap=6, aspect=1)
    g.map(sns.distplot, "lsnr", kde=True, hist=True);
def rssis_grid(data):
    g = sns.FacetGrid(data, col="cluster", col_wrap=6, aspect=1)
    g.map(sns.distplot, "rssis", kde=False, hist=True)

