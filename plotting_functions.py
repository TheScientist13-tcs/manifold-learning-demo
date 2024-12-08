import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def create_2d_meshgrid(X):
    h_min, h_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    v_min, v_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(h_min, h_max, 0.01), np.arange(v_min, v_max, 0.01))
    return (xx, yy)


def create_3d_meshgrid(X):
    h_min, h_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    v_min, v_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    g_min, g_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy, zz = np.meshgrid(
        np.arange(h_min, h_max, 0.01),
        np.arange(v_min, v_max, 0.01),
        np.arange(g_min, g_max, 0.01),
    )
    return (xx, yy, zz)


def plot_decision_boundary(
    classifier, X, y, title="Decision Boundary with Colored Regions"
):
    # Create a mesh grid
    xx, yy = create_2d_meshgrid(X)

    # Predict on the grid
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    return fig


def _reshape_list(ls: list, nrows, ncols):
    try:
        ls = np.array(ls)
        ls = ls.reshape(nrows, ncols)
        return ls
    except ValueError as e:
        print(f"Error: {e}")


def plot_multiclassifiers(
    classifiers: list,
    X,
    y,
    plt_nrows,
    plt_ncols,
    titles,
    figsize=(10, 4),
    xlabel="feature 1",
    ylabel="feature 2",
):

    xx, yy = create_2d_meshgrid(X)

    fig, ax = plt.subplots(nrows=plt_nrows, ncols=plt_ncols, figsize=figsize)
    classifiers_reshaped = _reshape_list(classifiers, plt_nrows, plt_ncols)

    ct = 0
    legend = False
    for i in range(plt_nrows):
        for j in range(plt_ncols):
            if ct == len(classifiers) - 1:
                legend = True
            y_pred = classifiers_reshaped[i][j].predict(np.c_[xx.ravel(), yy.ravel()])
            y_pred = y_pred.reshape(xx.shape)

            if plt_nrows == 1:
                ax[j].contourf(xx, yy, y_pred, alpha=0.5)
                sns.scatterplot(
                    x=X[:, 0],
                    y=X[:, 1],
                    hue=y,
                    palette="Paired",
                    edgecolor="k",
                    ax=ax[j],
                    legend=legend,
                )
                # ax[j].scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o")
                ax[j].set_title(titles[j])
                ax[j].set_xlabel(xlabel)
                ax[j].set_ylabel(ylabel)
            elif plt_nrows > 1:
                ax[i, j].contourf(xx, yy, y_pred, alpha=0.5)
                sns.scatterplot(
                    x=X[:, 0],
                    y=X[:, 1],
                    hue=y,
                    palette="Paired",
                    edgecolor="k",
                    ax=ax[i, j],
                    legend=legend,
                )
                # ax[i, j].scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o")
                ax[i, j].set_title(titles[ct])
                ax[i, j].set_xlabel(xlabel)
                ax[i, j].set_ylabel(ylabel)
            ct += 1
    plt.legend(bbox_to_anchor=(1.4, 1), loc="upper right", borderaxespad=0.0)
    plt.tight_layout()


def plot3d_multiclassifiers(
    classifiers: list,
    X,
    y,
    plt_nrows,
    plt_ncols,
    titles,
    figsize=(10, 4),
    xlabel="feature 1",
    ylabel="feature 2",
    zlabel="feature 3",
):

    xx, yy, zz = create_3d_meshgrid(X)

    fig, ax = plt.subplots(nrows=plt_nrows, ncols=plt_ncols, figsize=figsize)
    classifiers_reshaped = _reshape_list(classifiers, plt_nrows, plt_ncols)

    ct = 0
    legend = False
    for i in range(plt_nrows):
        for j in range(plt_ncols):
            if ct == len(classifiers) - 1:
                legend = True
            y_pred = classifiers_reshaped[i][j].predict(
                np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
            )
            print(f"Sanity Check: {y_pred.shape}")
            y_pred = y_pred.reshape(xx.shape)

            if plt_nrows == 1:
                ax[j].contourf(xx, yy, y_pred, alpha=0.5)
                sns.scatterplot(
                    x=X[:, 0],
                    y=X[:, 1],
                    hue=y,
                    palette="Paired",
                    edgecolor="k",
                    ax=ax[j],
                    legend=legend,
                )
                ax[j].set_title(titles[j])
                ax[j].set_xlabel("PC 1")
                ax[j].set_xlabel("PC 2")
            elif plt_nrows > 1:
                ax[i, j].plot_surface(xx, yy, zz, alpha=0.5)
                sns.scatterplot(
                    x=X[:, 0],
                    y=X[:, 1],
                    hue=y,
                    palette="Paired",
                    edgecolor="k",
                    ax=ax[i, j],
                    legend=legend,
                )
                ax[i, j].set_title(titles[ct])
                ax[i, j].set_xlabel("PC 1")
                ax[i, j].set_xlabel("PC 2")
            ct += 1
    plt.legend(bbox_to_anchor=(1.4, 1), loc="upper right", borderaxespad=0.0)
    plt.tight_layout()
