from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn import metrics
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, MeanShift, SpectralClustering, AgglomerativeClustering


class SklearnUtils:

    @staticmethod
    def knn_classifier_digits_dataset():
        digits = load_digits()
        x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                            random_state=11, test_size=0.20)
        knn = KNeighborsClassifier()
        knn.fit(x_train, y_train)
        predicted = knn.predict(X=x_test)
        expected = y_test
        wrong_predictions = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
        print('wrong predictions:', wrong_predictions)
        print(f'Accuracy: {knn.score(x_test, y_test):.2%}')
        confusion = confusion_matrix(y_true=expected, y_pred=predicted)
        names = [str(digit) for digit in digits.target_names]
        print(classification_report(expected, predicted, target_names=names))
        confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))
        axes = sns.heatmap(confusion_df, annot=True, cmap='nipy_spectral_r')
        plt.axes = axes
        plt.show()

        # region K FOLD CROSS VALIDATION
        k_fold = KFold(n_splits=10, random_state=11, shuffle=True)
        scores = cross_val_score(estimator=knn, X=digits.data, y=digits.target, cv=k_fold)
        print(f'cross validation scores:\n{scores}')
        print(f'Mean accuracy: {scores.mean():.2%}')
        print(f'Accuracy standard deviation: {scores.std():.2f}')
        # endregion

        # region MULTIPLE MODEL
        estimators = {
            'KNeighborsClassifier': knn,
            'SVC': SVC(gamma='scale'),
            'GaussianNB': GaussianNB()
        }
        for estimator_name, estimator_object in estimators.items():
            k_fold_multiple = KFold(n_splits=10, random_state=11, shuffle=True)
            scores = cross_val_score(estimator=estimator_object,
                                     X=digits.data, y=digits.target, cv=k_fold_multiple)
            print(f'{estimator_name:>20}: ' +
                  f'mean accuracy={scores.mean():.2%}; ' +
                  f'standard deviation={scores.std():.2%}')
        # end region

    @staticmethod
    def linear_regression():
        nyc = pd.read_csv(Path('resources/nyc.csv'))
        nyc.columns = ['Date', 'Temperature', 'Anomaly']
        nyc.Date = nyc.Date.floordiv(100)
        x_train, x_test, y_train, y_test = train_test_split(nyc.Date.values.reshape(-1, 1),
                                                            nyc.Temperature.values, random_state=11)

        linear_regression = LinearRegression()
        linear_regression.fit(X=x_train, y=y_train)
        predicted = linear_regression.predict(x_test)
        expected = y_train
        for p, e in zip(predicted[::2], expected[::2]):
            print(f'predicted: {p:.2f}, expected: {e:.2f}')

        predict = (lambda pre: linear_regression.coef_ * pre + linear_regression.intercept_)
        print('2022 prediction:', predict(2022))

        # create regression line
        x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
        y = predict(x)
        plt.plot(x, y)

        axes = sns.scatterplot(data=nyc, x='Date', y='Temperature',
                               hue='Temperature', palette='winter', legend=False)

        axes.set_ylim(10, 50)
        plt.axes = axes
        plt.show()

    @staticmethod
    def linear_regression_multiple():
        california_housing_data = fetch_california_housing()
        # print(california_housing_data.DESCR)
        # print(california_housing_data.feature_names)
        pd.set_option('precision', 4)
        # max number of columns to display when output data frame
        pd.set_option('max_columns', 9)
        # auto detects display width when formatting string representation
        pd.set_option('display.width', None)
        california_housing_df = pd.DataFrame(california_housing_data.data,
                                             columns=california_housing_data.feature_names)
        california_housing_df['MedHouseValue'] = pd.Series(california_housing_data.target)
        # print(california_housing_df.head())
        # print(california_housing_df.describe())
        sample_california_housing_df = california_housing_df.sample(frac=0.1, random_state=17)
        sns.set(font_scale=2)
        sns.set_style('whitegrid')
        # create matplotlib figures for each feature
        """
        for feature in california_housing_data.feature_names:
            plt.figure(figsize=(16, 9))
            sns.scatterplot(data=sample_california_housing_df, x=feature,
                            y='MedHouseValue', hue='MedHouseValue',
                            palette='cool', legend=False)

        plt.show()
        """
        x_train, x_test, y_train, y_test = train_test_split(california_housing_data.data,
                                                            california_housing_data.target,
                                                            random_state=11)
        linear_regression = LinearRegression()
        linear_regression.fit(X=x_train, y=y_train)
        for i, name in enumerate(california_housing_data.feature_names):
            print(f'{name:>10} {linear_regression.coef_[i]}')

        print(f'intercept: {linear_regression.intercept_}')
        predicted = linear_regression.predict(X=x_test)
        expected = y_test
        df = pd.DataFrame()
        df['Expected'] = pd.Series(expected)
        df['Predicted'] = pd.Series(predicted)
        plt.figure(figsize=(9, 9))
        axes = sns.scatterplot(data=df, x='Expected', y='Predicted',
                               hue='Predicted', palette='cool', legend=False)
        start = min(expected.min(), predicted.min())
        end = max(expected.max(), predicted.max())
        axes.set_xlim(start, end)
        # line of the perfect prediction not regression line,  k-- : color black and dashes
        plt.plot([start, end], [start, end], 'k--')
        # plt.show()

        r2_score = metrics.r2_score(expected, predicted)
        mean_squared_error = metrics.mean_squared_error(expected, predicted)
        # r2 score is between 0-1, 1 is the best
        print(f'R2 Score: {r2_score}')
        print(f'Mean Square Error: {mean_squared_error}')

        estimators = {
            'LinearRegression': linear_regression,
            'ElasticNet': ElasticNet(),
            'Lasso': Lasso(),
            'Ridge': Ridge()
        }

        print('-----------SCORES----------')
        for estimator_name, estimator_object in estimators.items():
            kfold = KFold(n_splits=10, random_state=11, shuffle=True)
            scores = cross_val_score(estimator=estimator_object,
                                     X=california_housing_data.data,
                                     y=california_housing_data.target,
                                     cv=kfold,
                                     scoring='r2')
            print(f'{estimator_name:>16}: ' + f'mean of r2 scores={scores.mean():.3f}')

    @staticmethod
    def dimensionality_reduction():
        digits = load_digits()
        tsne = TSNE(n_components=2, random_state=11)
        reduced_data = tsne.fit_transform(digits.data)
        dots = plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                    c=digits.target, cmap=plt.cm.get_cmap('nipy_spectral_r', 10))
        plt.colorbar(dots)
        plt.show()

    @staticmethod
    def k_means_clustering():
        iris = load_iris()
        pd.set_option('max_columns', 5)
        pd.set_option('display.width', None)
        pd.set_option('precision', 2)
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df['species'] = [iris.target_names[i] for i in iris.target]
        # print(iris_df)
        # sns.set(font_scale=1.1)
        # sns.set_style('whitegrid')
        # grid = sns.pairplot(data=iris_df, vars=iris_df.columns[0:4], hue='species')
        # plt.show()
        kmeans = KMeans(n_clusters=3, random_state=11)
        kmeans.fit(iris.data)
        pca = PCA(n_components=2, random_state=11)
        pca.fit(iris.data)
        iris_pca = pca.transform(iris.data)
        print(iris.data.shape)
        print(iris_pca.shape)
        iris_pca_df = pd.DataFrame(iris_pca, columns=['Component1', 'Component2'])
        iris_pca_df['species'] = iris_df.species
        axes = sns.scatterplot(data=iris_pca_df, x='Component1',
                               y='Component2', hue='species', legend='brief',
                               palette='cool')

        iris_centers = pca.transform(kmeans.cluster_centers_)
        # [:, 0] and [:, 1] refers to first column and second column
        # dots = plt.scatter(iris_centers[:, 0], iris_centers[:, 1], s=100, c='k')
        # plt.show()
        estimators = {
            'KMeans': kmeans,
            'DBSCAN': DBSCAN(),
            'MeanShift': MeanShift(),
            'SpectralClustering': SpectralClustering(n_clusters=3),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=3)}

        for name, estimator in estimators.items():
            estimator.fit(iris.data)
            print(f'\n{name}:')
            for i in range(0, 101, 50):
                labels, counts = np.unique(
                    estimator.labels_[i:i + 50], return_counts=True)
                print(f'{i}-{i+50}:')
                for label, count in zip(labels, counts):
                    print(f' label={label}, count={count}')









