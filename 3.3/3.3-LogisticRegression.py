import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def J_cost(X, y, beta):
    '''
    :param X:  sample array, shape(n_samples, n_features)
    :param y: array-like, shape (n_samples,)
    :param beta: the beta in formula 3.27 , shape(n_features + 1, ) or (n_features + 1, 1)
    :return: the result of formula 3.27
    '''
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)

    Lbeta = -y * np.dot(X_hat, beta) + np.log(1 + np.exp(np.dot(X_hat, beta)))

    return Lbeta.sum()


def gradient(X, y, beta):
    '''
    compute the first derivative of J(i.e. formula 3.27) with respect to beta      i.e. formula 3.30
    ----------------------------------
    :param X: sample array, shape(n_samples, n_features)
    :param y: array-like, shape (n_samples,)
    :param beta: the beta in formula 3.27 , shape(n_features + 1, ) or (n_features + 1, 1)
    :return:
    '''
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    p1 = sigmoid(np.dot(X_hat, beta))

    gra = (-X_hat * (y - p1)).sum(0)

    return gra.reshape(-1, 1)


def update_parameters_gradDesc(X, y, learning_rate = 1.2, num_iterations = 100, print_cost = False):
    '''
    update parameters with gradient descent method
    --------------------------------------------
    :param beta:
    :param grad:
    :param learning_rate:
    :return:
    '''
    m, n = X.shape
    beta = initialize_beta(n)
    for i in range(num_iterations):

        grad = gradient(X, y, beta)
        beta = beta - learning_rate * grad

        if (i % 10 == 0) & print_cost:
            print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))

    return beta

def initialize_beta(n):
    beta = np.random.randn(n + 1, 1) * 0.5 + 1
    return beta

def predict(X, beta):
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    p1 = sigmoid(np.dot(X_hat, beta))

    p1[p1 >= 0.5] = 1
    p1[p1 < 0.5] = 0

    return p1


if __name__ == '__main__':
    data_path = r'H:\MachineLearningTraining\MachineLearning\3.3\watermelon3_0_Ch.csv'
    #
    data = pd.read_csv(data_path).values

    is_good = data[:, 9] == '是'
    is_bad = data[:, 9] == '否'

    X = data[:, 7:9].astype(float)
    y = data[:, 9]

    y[y == '是'] = 1
    y[y == '否'] = 0
    y = y.astype(int)

    plt.scatter(data[:, 7][is_good], data[:, 8][is_good], c='k', marker='o')
    plt.scatter(data[:, 7][is_bad], data[:, 8][is_bad], c='r', marker='x')

    plt.xlabel('密度')
    plt.ylabel('含糖量')

    beta = update_parameters_gradDesc(X, y, learning_rate=0.3, num_iterations=1000,print_cost=True)
    w1, w2, intercept = beta
    Predict1 = predict(X,beta)
    Predict1 = Predict1.reshape(1, -1)
    print(Predict1)
    x1 = np.linspace(0, 1)
    y1 = -(w1 * x1 + intercept) / w2

    ax1, = plt.plot(x1, y1, label=r'my_logistic_gradDesc')

    lr = linear_model.LogisticRegression(solver='lbfgs', C=1000)  # C越大表示正则化程度越低。
    lr.fit(X, y)
    Predict2 = lr.predict(X)
    print(Predict2)
    Score2 = lr.score(X,y)
    print(Score2)

    lr_beta = np.c_[lr.coef_, lr.intercept_]
    print(J_cost(X, y, lr_beta))

    w1_sk, w2_sk = lr.coef_[0, :]

    x2 = np.linspace(0, 1)
    y2 = -(w1_sk * x2 + lr.intercept_) / w2

    ax2, = plt.plot(x2, y2, label=r'sklearn_logistic')

    plt.legend(loc='upper right')
    plt.show()
