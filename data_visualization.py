import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# visulaize the important characteristics of the dataset
import matplotlib.pyplot as plt


def process_data(filename, label, remove_columns, drop):
    pd.set_option('display.max_columns', None)
    # step 1: download the data
    df = pd.read_csv(filename)
    if drop:
        df = df.dropna(axis=0, how='any')
    # print df
    num_rows = df.shape[0]
    # step 2: remove useless data
    # count the number of missing elements (NaN) in each column
    counter_nan = df.isnull().sum()
    counter_without_nan = counter_nan[counter_nan == 0]
    # remove the columns with missing elements
    df = df[counter_without_nan.keys()]
    # remove the first 7 columns which contain no discriminative information
    df = df.ix[:, :]
    # df = df.drop(columns=remove_columns)
    # the list of columns (the last column is the class label)
    columns = df.columns
    # print columns

    # step 3: get class labels y and then encode it into number
    # get class label data
    # print type(label)
    y = df[label].values
    # print y
    y = y.reshape([y.shape[0]])
    # encode the class label
    # class_labels = np.unique(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # step 4: get features (x) and scale the features
    # get x and convert it to numpy array
    df = df.drop(columns=label).apply(LabelEncoder().fit_transform)
    standard_scaler = StandardScaler()
    x = df.ix[:, :].values
    x_std = standard_scaler.fit_transform(x)
    x_std = np.delete(x_std, remove_columns, axis=1)
    # # step 5: split the data into training set and test set
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x_std, y, test_size=test_percentage, random_state=0)

    # return x_train, x_test, y_train, y_test

    return x_std, y


def perform_TSNE(x_test, y_test):
    # t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0)
    x_test_2d = tsne.fit_transform(x_test)

    # scatter plot the sample points among 5 classes
    markers = ('s', 'd', 'o', '^', 'v')
    color_map = {0: 'red', 1: 'blue', 2: 'lightgreen', 3: 'purple', 4: 'cyan'}
    plt.figure()
    for idx, cl in enumerate(np.unique(y_test)):
        plt.scatter(x=x_test_2d[y_test == cl, 0], y=x_test_2d[
            y_test == cl, 1], c=color_map[idx], marker=markers[idx], label=cl)
    plt.xlabel('X in t-SNE')
    plt.ylabel('Y in t-SNE')
    plt.legend(loc='upper left')
    plt.title('t-SNE visualization of test data')
    plt.show()
