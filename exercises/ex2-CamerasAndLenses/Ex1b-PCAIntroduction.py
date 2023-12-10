import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import datasets
import seaborn as sns
import pandas as pd


# https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
def pca_iris_example():
    np.random.seed(5)

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()

    ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    ax.set_position([0, 0, 0.95, 1])

    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
        ax.text3D(
            X[y == label, 0].mean(),
            X[y == label, 1].mean() + 1.5,
            X[y == label, 2].mean(),
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
        )
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()


def explorative_analysis_iris_data():
    in_dir = "data/"
    txt_name = "irisdata.txt"

    iris_data = np.loadtxt(in_dir + txt_name, comments="%")
    # x is a column matrix with 50 rows and 4 columns
    x = iris_data[0:50, 0:4]
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")

    sep_l = x[:, 0]
    sep_w = x[:, 1]
    pet_l = x[:, 2]
    pet_w = x[:, 3]

    # Use ddof = 1 to make an unbiased estimate
    var_sep_l = sep_l.var(ddof=1)
    var_sep_w = sep_w.var(ddof=1)
    var_pet_l = pet_l.var(ddof=1)
    var_pet_w = pet_w.var(ddof=1)
    print(f"Sepal length, width variance: {var_sep_l}, {var_sep_w}. "
          f"Petal length, width variance: {var_pet_l}, {var_pet_w}")

    cov_sep_l_sep_w = np.dot(sep_l, sep_w) / (n_obs - 1)
    # cov_sep_l_sep_w = np.cov(sep_l, sep_w)  # gives a different result
    print(f"Covariance between sepal length and width: {cov_sep_l_sep_w}")
    cov_sep_l_pet_l = np.dot(sep_l, pet_l) / (n_obs - 1)
    print(f"Covariance between sepal and petal length: {cov_sep_l_pet_l}")

    # https://regenerativetoday.com/pairplot-and-pairgrid-in-python/
    # For some kind of reason this plt.figure needs to be here
    plt.figure()
    # Transform the data into a Pandas dataframe
    d = pd.DataFrame(x, columns=['Sepal length', 'Sepal width',
                                 'Petal length', 'Petal width'])
    sns.pairplot(d)
    plt.show()


def pca_on_iris_data():
    in_dir = "data/"
    txt_name = "irisdata.txt"

    iris_data = np.loadtxt(in_dir + txt_name, comments="%")
    # x is a matrix with 50 rows and 4 columns
    x = iris_data[0:50, 0:4]
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")

    mn = np.mean(x, axis=0)
    # print(mn)
    data = x - mn
    # print(data)
    c_x = np.cov(data.T)
    c_x_2 = np.matmul(data.T,  data) / (n_obs - 1)

    # print(Cx)
    values, vectors = np.linalg.eig(c_x)
    # print(vectors)
    print(values)

    v_norm = values / values.sum() * 100
    print(v_norm)
    plt.plot(v_norm)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.ylim([0, 100])
    plt.show()

    # Project data
    pc_proj = vectors.T.dot(data.T)
    # print(p)
    plt.figure()
    d = pd.DataFrame(pc_proj.T)
    sns.pairplot(d)
    plt.show()

    # Compare with direct PCA
    # https://scikit-learn.org/stable/modules/decomposition.html#pca
    pca = decomposition.PCA()
    pca.fit(x)
    plt.plot(pca.explained_variance_ratio_)
    plt.show()

    values_2 = pca.explained_variance_
    # This is transposed compared to "vectors"
    vectors_2 = pca.components_
    # print(values_2)

    # This is transposed compared to "pc_proj" above
    data_transform = pca.transform(data)


if __name__ == '__main__':
    # pca_iris_example()
    pca_on_iris_data()
    # explorative_analysis_iris_data()
