import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma, trapz

#Fill the NaN values and drop the blank rows
for k in range(5):
#provide absolute path to data folder before /CGMSeriesLunchPat{0}.csv
    df = pd.read_csv("/CGMSeriesLunchPat{0}.csv".format(k+1))
    final_clean = []
    for i in range(0,df.shape[0]):
        row=df.iloc[[i]]
        row_arr=row.to_numpy()
        record=pd.Series(row_arr[0])
        processed=record.interpolate(method='spline', order=2,limit_direction='both')
        final_clean.append(np.floor(np.abs((processed.to_numpy()))))

    final_clean=np.asarray(final_clean)
    df_cleaned=pd.DataFrame(final_clean).dropna()
    df_cleaned.to_csv('/data/CGMSeriesLunchPat{0}_clean.csv'.format(k+1))

#calculate moving average
def movingAvg(data, size):
        weights = np.repeat(1.0, size)/size
        ma = np.convolve(data, weights, 'valid')
        return ma

for k in range(5):
#provide absolute path to data folder before /CGMSeriesLunchPat{0}.csv
    df = pd.read_csv("/CGMSeriesLunchPat{0}_clean.csv".format(k+1))

    #creating feature matrix using four time-series features for each patient
    features=[]
    for i in range(0,df.shape[0]):
        row = df.iloc[[i]]
        row_arr = row.to_numpy()
        output = []
        for j in range(len(row_arr[0]) - 1):
            areaUnderTheCurve = trapz([row_arr[0][j], row_arr[0][j+1]], dx=5)
            output.append(areaUnderTheCurve)

        for j in range(len(row_arr[0]) - 1):
            velocity = (row_arr[0][j+1] - row_arr[0][j])/5
            output.append(velocity)

        rfft=np.fft.rfft(row_arr[0])
        rfft_log=np.log(np.abs(rfft) ** 2)

        output.extend(movingAvg(row_arr[0],2))

        output.extend(rfft_log)
        features.append(output)


    df_feature=pd.DataFrame(features)
    #provide absolute path to data folder before /CGMSeriesLunchPat{0}.csv
    df_feature.to_csv('/feature_matrix{0}.csv'.format(k+1))

#PCA Analysis
def pca_analysis():
    for k in range(5):
    #provide absolute path to data folder before /CGMSeriesLunchPat{0}.csv
        df = pd.read_csv("/feature_matrix{0}.csv".format(k+1))
        scaled_df=StandardScaler().fit_transform(df)
        pca = PCA(n_components=5)
        principalComponents = pca.fit_transform(scaled_df)
        principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5'])
        variance=pca.explained_variance_

        #Plot the Variance
        with plt.style.context('bmh'):
            plt.figure(figsize=(6, 4))
            plt.bar(range(5), pca.explained_variance_, alpha=0.5, align='center',label='individual explained variance')
            plt.ylabel('Explained variance ratio')
            plt.xlabel('Principal components')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()

        #Plot the eigen vectors
        index=range(1,pca.components_.shape[1]+1)
        x=pca.components_[4]
        plt.bar(index, x)
        plt.xlabel('index', fontsize=5)
        plt.ylabel('vectors', fontsize=5)
        plt.xticks(index, x, fontsize=5, rotation=30)
        plt.title('Eigen Vector plotted on the graph')
        plt.show()

        #Top five features
        pca = PCA(n_components=5)
        principalComponents = pca.fit_transform(scaled_df)
        s = pca.explained_variance_
        vt = pca.components_
        for i in vt:
            i = i.sort()
        plt.plot(vt[:,-5:])
        plt.xlabel('PCA Components')
        plt.ylabel('Weights for Vectors')
        plt.show()

#execute pca analysis
pca_analysis()

