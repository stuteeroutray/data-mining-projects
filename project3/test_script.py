import pandas as pd
import csv
from csv import reader
import numpy as np
from numpy import trapz
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import itertools
from collections import defaultdict
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
import joblib


def clean_meal_data_extract_features():
    df = pd.read_csv("input/input.csv", header=None, sep='\n')
    df = df[0].str.split(',', expand=True)
    df.to_csv('output/test/clean_input.csv', header=False, index=False)
    df = pd.read_csv('output/test/clean_input.csv', header=None)
    df.replace(np.nan, df.mean(axis=0), inplace=True)
    df.replace(np.nan, df.mean(axis=1), inplace=True)
    df.to_csv('output/test/clean_input.csv', header=False, index=False)

    def moving_avg(data, size):
        weights = np.repeat(1.0, size) / size
        ma = np.convolve(data, weights, 'valid')
        return ma

    feature_set = []
    df_read = pd.read_csv("output/test/clean_input.csv", header=None)
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
    f_mat = pd.DataFrame(data=scaled_df)

    pca = PCA(n_components=1)
    pca.fit_transform(scaled_df)
    vt = pca.components_
    components = vt.transpose()
    f_mat_new = np.dot(f_mat, components)
    data = pd.DataFrame(data=f_mat_new)
    data.to_csv('output/test/pca_input.csv', index=False, header=False)

    loaded_model = joblib.load('output/model/knn_model.pkl')
    predicted = loaded_model.predict(data)
    df = pd.DataFrame(predicted)
    df.to_csv('output/test/knn_predicted.csv', header=False, index=False)


def clustering():
    pca_data = pd.read_csv("output/test/pca_input.csv", header=None)
    db_default = DBSCAN(eps=.39, min_samples=3).fit(pca_data)
    labels = db_default.labels_
    dbscan_labels = []
    dbscan_labels_file = pd.DataFrame()
    for i in range(0, pca_data.shape[0]):
        data_col = pca_data.iloc[[i]]
        new_row = [data_col.to_numpy(), labels[i]]

        dbscan_labels.append(new_row)
    df = pd.DataFrame(dbscan_labels)
    dbscan_labels_file = pd.concat([dbscan_labels_file, df])
    dbscan_labels_file.to_csv('output/test/test_dbscan_results.csv', header=False, index=False)

    kmeans = KMeans(n_clusters=8)
    kmeans.fit(pca_data)
    y_kmeans = kmeans.predict(pca_data)
    kmeans_labels = []
    kmeans_labels_file = pd.DataFrame()
    for i in range(0, pca_data.shape[0]):
        data_col = pca_data.iloc[[i]]
        new_row = [data_col.to_numpy(), y_kmeans[i]]

        kmeans_labels.append(new_row)
    df = pd.DataFrame(kmeans_labels)
    kmeans_labels_file = pd.concat([kmeans_labels_file, df])
    kmeans_labels_file.to_csv('output/test/test_kmeans_results.csv', header=False, index=False)

    dbscan_res = defaultdict(list)
    kmeans_res = defaultdict(list)
    f1 = open("output/test/test_dbscan_results.csv")
    f2 = open("output/test/test_kmeans_results.csv")
    f3 = open("output/test/knn_predicted.csv")

    kdict = {}
    ddict = {}

    csv_f1 = csv.reader(f1)
    csv_f2 = csv.reader(f2)
    csv_f3 = csv.reader(f3)
    for row1,row2,row3 in itertools.zip_longest(csv_f1, csv_f2, csv_f3):
        if row1[1] != '-1':
            dbscan_res[row1[1]].append(row3[0])
        if row2[1] != '-1':
            kmeans_res[row2[1]].append(row3[0])

    for i in dbscan_res.keys():
        values = dbscan_res[list(dbscan_res.keys())[int(i)]]
        test_list = Counter(values)
        class_label = test_list.most_common(1)[0][0]
        ddict[i] = class_label

    for i in kmeans_res.keys():
        values = kmeans_res[list(kmeans_res.keys())[int(i)]]
        test_list = Counter(values)
        class_label = test_list.most_common(1)[0][0]
        kdict[i] = class_label

    f1 = open("output/test/test_dbscan_results.csv")
    f2 = open("output/test/test_kmeans_results.csv")
    csv_f1 = csv.reader(f1)
    csv_f2 = csv.reader(f2)
    result = []
    with open('output/results/labels.csv', mode='w') as label_file:
        label_writer = csv.writer(label_file, delimiter=',')
        for row1,row2 in itertools.zip_longest(csv_f1, csv_f2):
            if row1[1] == '-1':
                r1 = '-'
            else:
                r1 = ddict[row1[1]]

            if row2[1] == '-1':
                r2 = '-'
            else:
                r2 = kdict[row2[1]]
            label_writer.writerow([r1,r2])


def main():
    clean_meal_data_extract_features()
    clustering()


if __name__ == '__main__':
    main()