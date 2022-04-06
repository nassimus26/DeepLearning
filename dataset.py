import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

'''
 * @author Nassim MOUALEK
 * @since 06/12/2020
'''

def generateDataset(rows): # 2 features and one target (sick or not)
    rows_div_8 = int(rows/8)
    sick = np.random.randn(rows_div_8, 2) + np.array([-2, -2])
    sick_2 = np.random.randn(rows_div_8, 2) + np.array([2, 2])
    healthy = np.random.randn(rows_div_8, 2) + np.array([-2, 2])
    healthy_2 = np.random.randn(rows_div_8, 2) + np.array([2, -2])
    features = np.vstack([sick, sick_2, healthy, healthy_2])
    targets = np.concatenate((np.zeros(rows_div_8*2),
                              np.zeros(rows_div_8*2) + 1))
    targets = targets.reshape(-1, 1)
    return features, targets

def extendDataset(features): # extend the Features (X, Y) with (X°2, Y°2, X1*X2, sin(X1), sin(X2)
    features = np.concatenate([features, np.square(features[:, 0])[:, None],
                               np.square(features[:, 1])[:, None],
                               np.multiply(features[:, 0], features[:, 1])[:, None],
                               np.sin(features[:, 0])[:, None],
                               np.sin(features[:, 1])[:, None]], -1)
    return features


def getSprialDataset(rows):
    theta = np.sqrt(np.random.rand(rows))*2*pi # np.linspace(0,2*pi,100)

    r_a = 2 * theta + pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + np.random.randn(rows,2)

    r_b = -2 * theta - pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = data_b + np.random.randn(rows,2)

    res_a = np.append(x_a, np.zeros((rows,1)), axis=1)
    res_b = np.append(x_b, np.ones((rows,1)), axis=1)

    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)
    features = res[:, 0:2]
    targets = res[:, 2]
    return features, targets

def show(features, targets):
    colors = np.where(targets.reshape(-1), "green", "red")
    plt.scatter(features[:, 0], features[:, 1], c=colors)
    plt.draw()
    plt.pause(0.001)
    input("Press ENTER to terminate.")
    return