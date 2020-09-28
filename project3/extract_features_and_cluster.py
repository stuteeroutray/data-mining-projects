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


def assign_bins_to_carbohydrates():
    for k in range(5):
        with open('input/mealAmountData/mealAmountData{0}.csv'.format(k + 1), 'r') as read_obj:
            with open('output/labels/labels{0}.csv'.format(k + 1), 'w', newline='') as file:
                csv_reader = reader(read_obj)
                writer = csv.writer(file)
                for row in csv_reader:
                    if int(row[0]) == 0:
                        writer.writerow([0])
                    elif 0 < int(row[0]) <= 20:
                        writer.writerow([1])
                    elif 20 < int(row[0]) <= 40:
                        writer.writerow([2])
                    elif 40 < int(row[0]) <= 60:
                        writer.writerow([3])
                    elif 60 < int(row[0]) <= 90:
                        writer.writerow([4])
                    elif 80 < int(row[0]) <= 100:
                        writer.writerow([5])
                    elif 100 < int(row[0]) <= 120:
                        writer.writerow([6])
                    elif 120 < int(row[0]) <= 140:
                        writer.writerow([7])

    bins = pd.DataFrame()
    for k in range(5):
        labels = pd.read_csv("output/labels/labels{0}.csv".format(k + 1), header=None, sep='\n', nrows=50)
        bins = pd.concat([bins, labels])
        bins.to_csv('output/bins/bins.csv'.format(k + 1), header=False, index=False)


def clean_meal_data():
    for k in range(5):
        df = pd.read_csv("input/mealData/mealData{0}.csv".format(k + 1), header=None, sep='\n')
        df = df[0].str.split(',', expand=True)
        df.to_csv('output/mealData/mealData{0}.csv'.format(k + 1), header=False, index=False)
        df = pd.read_csv('output/mealData/mealData{0}.csv'.format(k + 1))
        df.replace(np.nan, df.mean(axis=0), inplace=True)
        df.replace(np.nan, df.mean(axis=1), inplace=True)
        # cleaned meal data files created in the project folder
        df.to_csv('output/mealData/mealData{0}.csv'.format(k + 1), header=False, index=False)


def moving_avg(data, size):
    weights = np.repeat(1.0, size) / size
    ma = np.convolve(data, weights, 'valid')
    return ma


def extract_features():
    features_meal = pd.DataFrame()
    for k in range(5):
        feature_set = []
        df = pd.read_csv("output/mealData/mealData{0}.csv".format(k + 1), header=None)
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
            result.extend(rfft_log)
            result.extend(moving_avg(row_arr[0], 3))

            feature_set.append(result)
        df_feature = pd.DataFrame(feature_set)
        features_meal = pd.concat([features_meal, df_feature])

    # consolidated feature matrix for meal data is created in the project folder
    features_meal.to_csv('output/pcaAndFeatureMatrix/feature_matrix_MealData.csv', header=False, index=False)

    df = pd.read_csv("output/pcaAndFeatureMatrix/feature_matrix_MealData.csv", header=None)
    scaled_df = StandardScaler().fit_transform(df)
    f_mat = pd.DataFrame(data=scaled_df)

    pca = PCA(n_components=1)
    pca.fit_transform(scaled_df)
    vt = pca.components_
    components = vt.transpose()
    f_mat_new = np.dot(f_mat, components)
    components_df_meal = pd.DataFrame(data=f_mat_new)
    knn_model(components_df_meal)
    components_df_meal.to_csv('output/pcaAndFeatureMatrix/pca_components_meal.csv', index=False, header=False)


def assign_bins_to_pca_data():
    label_assigned_meal_data = pd.DataFrame()
    labelled_data = []
    meal_data = pd.read_csv("output/pcaAndFeatureMatrix/pca_components_meal.csv", header=None)
    labels = pd.read_csv("output/bins/bins.csv", header=None)
    for i in range(0, meal_data.shape[0]):
        data_col = meal_data.iloc[[i]]
        label_col = labels.iloc[[i]]
        new_row = [data_col.to_numpy(), label_col.to_numpy()]

        labelled_data.append(new_row)
    df = pd.DataFrame(labelled_data)
    label_assigned_meal_data = pd.concat([label_assigned_meal_data, df])
    label_assigned_meal_data.to_csv('output/labels/labelled_data.csv', header=False, index=False)
    with open('output/labels/labelled_data.csv', 'r') as file:
        file_data = file.read()
        file_data = file_data.replace('[', '')
        file_data = file_data.replace(']', '')
        file_data = file_data.replace('\'', '')
        file_data = file_data.replace('"', '')
    with open('output/labels/labelled_data.csv', 'w') as file:
        file.write(file_data)


def clustering():
    pca_data = pd.read_csv("output/pcaAndFeatureMatrix/pca_components_meal.csv", header=None)
    db_default = DBSCAN(eps=.0227, min_samples=3).fit(pca_data)
    labels = db_default.labels_
    dbscan_labels = []
    dbscan_labels_file = pd.DataFrame()
    for i in range(0, pca_data.shape[0]):
        data_col = pca_data.iloc[[i]]
        new_row = [data_col.to_numpy(), labels[i]]

        dbscan_labels.append(new_row)
    df = pd.DataFrame(dbscan_labels)
    dbscan_labels_file = pd.concat([dbscan_labels_file, df])
    dbscan_labels_file.to_csv('output/clusters/dbscan_results.csv', header=False, index=False)

    kmeans = KMeans(n_clusters=8)
    kmeans.fit(pca_data)
    centroids = kmeans.cluster_centers_
    y_kmeans = kmeans.predict(pca_data)
    kmeans_labels = []
    kmeans_labels_file = pd.DataFrame()
    for i in range(0, pca_data.shape[0]):
        data_col = pca_data.iloc[[i]]
        new_row = [data_col.to_numpy(), y_kmeans[i]]

        kmeans_labels.append(new_row)
    df = pd.DataFrame(kmeans_labels)
    kmeans_labels_file = pd.concat([kmeans_labels_file, df])
    kmeans_labels_file.to_csv('output/clusters/kmeans_results.csv', header=False, index=False)

    sse = 0
    # calculate square of Euclidean distance of each point from its cluster center
    for i in range(0, pca_data.shape[0]):
        curr_center = centroids[y_kmeans[i]]
        sse += (pca_data.iloc[[i]].to_numpy() - curr_center[0]) ** 2

    print('SSE for K-Means:', sse)


def classification_error_calculation():
    dbscan_res = defaultdict(list)
    kmeans_res = defaultdict(list)
    f1 = open("output/clusters/dbscan_results.csv")
    f2 = open("output/clusters/kmeans_results.csv")
    f3 = open("output/labels/labelled_data.csv")

    csv_f1 = csv.reader(f1)
    csv_f2 = csv.reader(f2)
    csv_f3 = csv.reader(f3)
    for row1,row2,row3 in itertools.zip_longest(csv_f1, csv_f2, csv_f3):
        if row1[1] != '-1':
            dbscan_res[row1[1]].append(row3[1])
        if row2[1] != '-1':
            kmeans_res[row2[1]].append(row3[1])

    fp_classified = 0
    total_elements = 0
    for i in dbscan_res.keys():
        values = dbscan_res[list(dbscan_res.keys())[int(i)]]
        total_elements+=len(values)
        test_list = Counter(values)
        class_label = test_list.most_common(1)[0][0]
        fp_classified += (len(values) - values.count(class_label))

    dbscan_classification_error = (fp_classified/total_elements) * 100
    print('DBSCAN Classification Error:',dbscan_classification_error,'%')

    fp_classified = 0
    total_elements = 0
    for i in kmeans_res.keys():
        values = kmeans_res[list(kmeans_res.keys())[int(i)]]
        total_elements += len(values)
        test_list = Counter(values)
        class_label = test_list.most_common(1)[0][0]
        fp_classified += (len(values) - values.count(class_label))

    kmeans_classification_error = (fp_classified/total_elements) * 100
    print('K-Means Classification Error:',kmeans_classification_error,'%')


def knn_model(features):
    model = KNeighborsClassifier(n_neighbors=4)
    label = pd.read_csv("output/bins/bins.csv", header=None)
    model.fit(features,label.to_numpy().ravel())
    joblib.dump(model, 'output/model/knn_model.pkl')


def main():
    assign_bins_to_carbohydrates()
    clean_meal_data()
    extract_features()
    assign_bins_to_pca_data()
    clustering()
    classification_error_calculation()


if __name__ == '__main__':
    main()
