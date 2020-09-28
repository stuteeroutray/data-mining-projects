import numpy as np
import pandas as pd
from numpy import trapz
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib


def clean_data(path):
    df = pd.read_csv(path, header=None, sep='\n')
    df = df[0].str.split(',', expand=True)
    df.to_csv('clean_data.csv', header=False, index=False)
    df = pd.read_csv('clean_data.csv')
    df.replace(np.nan, df.mean(axis=0), inplace=True)
    df.replace(np.nan, df.mean(axis=1), inplace=True)
    # cleaned input data created in project folder
    df.to_csv('clean_data.csv', header=False, index=False)
    return df


def feature_extraction():
    def moving_avg(data, size):
        weights = np.repeat(1.0, size) / size
        ma = np.convolve(data, weights, 'valid')
        return ma

    feature_set = []
    df_read = pd.read_csv('clean_data.csv')
    df = pd.DataFrame(df_read)
    for i in range(0, df.shape[0]):

        row = df.iloc[[i]]
        row_arr = row.to_numpy()
        result = []

        for j in range(len(row_arr[0]) - 1):
            calc_area = trapz([row_arr[0][j], row_arr[0][j + 1]], dx=5)
            result.append(calc_area)

        for j in range(len(row_arr[0]) - 1):
            calc_velocity = (row_arr[0][j + 1] - row_arr[0][j]) / 5
            result.append(calc_velocity)

        rfft = np.fft.rfft(row_arr[0])
        rfft_log = np.log(np.abs(rfft) ** 2 + 1)
        result.extend(moving_avg(row_arr[0], 2))
        result.extend(rfft_log)
        feature_set.append(result)
    df_feature = pd.DataFrame(feature_set)
    scaled_df = StandardScaler().fit_transform(df_feature)
    return pd.DataFrame(scaled_df)


def pca(input):
    scaled_df = StandardScaler().fit_transform(input)
    f_mat = pd.DataFrame(data=scaled_df)
    pca = PCA(n_components=4)
    pca.fit_transform(scaled_df)
    vt = pca.components_
    components = vt.transpose()
    f_mat_new = np.dot(f_mat, components)
    components_df_meal = pd.DataFrame(data=f_mat_new)
    return components_df_meal


def test_model(model_file, data):
    loaded_model = joblib.load(model_file)
    ypred = loaded_model.predict(data)
    return ypred


def main():
    # provide the input file path below
    input_path = "input.csv"
    clean_data(input_path)
    df = feature_extraction()
    df_pca = pca(df)

    input = df_pca.iloc[:, 0:4].to_numpy()
    ypred = test_model('model.pkl', input)

    print("Predicted Values: ", ypred)


if __name__ == '__main__':
    main()
