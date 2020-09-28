import pandas as pd
import numpy as np
from numpy import trapz
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib


def clean_meal_data():
    for k in range(5):
        df = pd.read_csv("MealNoMealData/mealData{0}.csv".format(k + 1), header=None, sep='\n')
        df = df[0].str.split(',', expand=True)
        df.to_csv('cleanMealData{0}.csv'.format(k + 1), header=False, index=False)
        df = pd.read_csv('cleanMealData{0}.csv'.format(k + 1))
        df.replace(np.nan, df.mean(axis=0), inplace=True)
        df.replace(np.nan, df.mean(axis=1), inplace=True)
        # cleaned meal data files created in the project folder
        df.to_csv('cleanMealData{0}.csv'.format(k + 1), header=False, index=False)


def clean_no_meal_data():
    for k in range(5):
        df = pd.read_csv("MealNoMealData/Nomeal{0}.csv".format(k + 1), header=None, sep='\n')
        df = df[0].str.split(',', expand=True)
        df.to_csv('cleanNoMeal{0}.csv'.format(k + 1), header=False, index=False)
        df = pd.read_csv('cleanNoMeal{0}.csv'.format(k + 1))
        df.replace(np.nan, df.mean(axis=0), inplace=True)
        df.replace(np.nan, df.mean(axis=1), inplace=True)
        # cleaned no meal data files created in the project folder
        df.to_csv('cleanNoMeal{0}.csv'.format(k + 1), header=False, index=False)


def moving_avg(data, size):
    weights = np.repeat(1.0, size) / size
    ma = np.convolve(data, weights, 'valid')
    return ma


def extract_meal_data_features():
    features_meal = pd.DataFrame()
    for k in range(5):
        feature_set = []
        df = pd.read_csv("cleanMealData{0}.csv".format(k + 1))
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
    features_meal.to_csv('feature_matrix_MealData.csv', header=False, index=False)


def extract_no_meal_data_features():
    features_no_meal = pd.DataFrame()
    for k in range(5):
        feature_set = []
        # cleaned data file is created in the project folder
        df = pd.read_csv("cleanNoMeal{0}.csv".format(k + 1))
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
        features_no_meal = pd.concat([features_no_meal, df_feature])
    # consolidated feature matrix for no meal data is created in the project folder
    features_no_meal.to_csv('feature_matrix_NoMealData.csv', header=False, index=False)


def pca():
    # pca for meal
    df = pd.read_csv("feature_matrix_MealData.csv", header=None)
    scaled_df = StandardScaler().fit_transform(df)
    f_mat = pd.DataFrame(data=scaled_df)
    pca = PCA(n_components=4)
    pca.fit_transform(scaled_df)
    vt = pca.components_
    components = vt.transpose()
    f_mat_new = np.dot(f_mat, components)
    components_df_meal = pd.DataFrame(data=f_mat_new)
    components_df_meal.to_csv('pca_components_meal.csv', index=False, header=False)

    # pca for no meal
    df = pd.read_csv("feature_matrix_NoMealData.csv", header=None)
    scaled_df = StandardScaler().fit_transform(df)
    f_mat = pd.DataFrame(data=scaled_df)
    pca = PCA(n_components=4)
    pca.fit_transform(scaled_df)
    vt = pca.components_
    components = vt.transpose()
    f_mat_new = np.dot(f_mat, components)
    components_df_meal = pd.DataFrame(data=f_mat_new)
    components_df_meal.to_csv('pca_components_no_meal.csv', index=False, header=False)


# creates .pickle file
def model_train(df_train):
    feature_set = df_train.iloc[:, 0:4].to_numpy()
    class_labels = df_train['class'].to_numpy()
    model = SVC(gamma='auto')
    model.fit(feature_set, class_labels)
    # pkl file created in the project folder
    joblib.dump(model, 'model.pkl')


def main():
    clean_meal_data()
    clean_no_meal_data()
    extract_meal_data_features()
    extract_no_meal_data_features()
    pca()

    # Training:
    df_meal = pd.read_csv('pca_components_meal.csv', header=None)
    df_no_meal = pd.read_csv('pca_components_no_meal.csv', header=None)
    label_meal = np.array([1] * len(df_meal))
    label_no_meal = np.array([0] * len(df_no_meal))
    df_meal['class'] = label_meal
    df_no_meal['class'] = label_no_meal
    df = pd.concat([df_meal, df_no_meal])
    model_train(df)


if __name__ == '__main__':
    main()
