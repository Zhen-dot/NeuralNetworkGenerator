from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import os


def load_data(path):
    li = []
    with open(path) as f:
        c = csv.reader(f, delimiter=',')
        for r in c:
            li.append(r)
    return li


def load_digits():
    tra = load_data('data/optdigits.tra')
    tra = {'data': np.asarray([x[:-1] for x in tra]).astype(np.float),
           'target': np.asarray([x[-1] for x in tra]).astype(np.int32)}
    tes = load_data('data/optdigits.tes')
    tes = {'data': np.asarray([x[:-1] for x in tes]).astype(np.float),
           'target': np.asarray([x[-1] for x in tes]).astype(np.int32)}

    return tra, tes


def visualise(mlp, title, ax):
    plt.axis('off')
    # get number of neurons in each layer
    n_neurons = [len(layer) for layer in mlp.coefs_]
    n_neurons.append(mlp.n_outputs_)

    # calculate the coordinates of each neuron on the graph
    y_range = [0, max(n_neurons)]
    loc_neurons = [[[l, (n + 1) * (y_range[1] / (layer + 1))] for n in range(layer)] for l, layer in
                   enumerate(n_neurons)]
    x_neurons = [x for layer in loc_neurons for x, y in layer]
    y_neurons = [y for layer in loc_neurons for x, y in layer]

    # identify the range of weights
    weight_range = [min([layer.min() for layer in mlp.coefs_]), max([layer.max() for layer in mlp.coefs_])]

    # prepare the figure
    ax.set_title(title, fontsize='x-small')
    # draw the neurons
    ax.scatter(x_neurons, y_neurons, s=5, zorder=5)
    # draw the connections with line width corresponds to the weight of the connection
    for l, layer in enumerate(mlp.coefs_):
        for i, neuron in enumerate(layer):
            for j, w in enumerate(neuron):
                ax.plot([loc_neurons[l][i][0], loc_neurons[l + 1][j][0]],
                        [loc_neurons[l][i][1], loc_neurons[l + 1][j][1]], 'grey',
                        linewidth=(w - weight_range[0]) / ((weight_range[1] - weight_range[0]) * 4) + 0.01)


def train(layer, x_train, x_test, y_train, y_test, **kwargs):
    fig = plt.figure()

    mlp = MLPClassifier(hidden_layer_sizes=layer, max_iter=10000)
    mlp.partial_fit(x_train, y_train, np.unique(y_train))
    visualise(mlp, "Before training", fig.add_subplot(121))

    mlp.fit(x_train, y_train)

    predictions = mlp.predict(x_test)
    score = "%.4f" % accuracy_score(y_test, predictions)

    visualise(mlp, "After training", fig.add_subplot(122))

    fig.suptitle(f'{layer}\nscore = {score}', fontsize='small')

    for k, v in kwargs.items():
        if k == 'save' and v:
            fig.savefig(f'{v}/{len(layer)}-{layer}.png', dpi=300)
            with open(f'{v}/{len(layer)}-{layer}.txt', 'w') as f:
                f.write(f'Layers - {layer}\n'
                        f'Iterations - {mlp.n_iter_}\n'
                        f'Score - {score}\n'
                        f'Classification report\n'
                        f'{classification_report(y_test, predictions)}')
                f.close()
        elif k == 'show' and v:
            plt.show()
        elif k == 'details' and v:
            print(confusion_matrix(y_test, predictions))
            print(classification_report(y_test, predictions))
            print(f'n iter {mlp.n_iter_}')
            print(f'coefs {mlp.coefs_}')
            print(f'intercepts {mlp.intercepts_}')

    plt.close(fig)
    return {'mlp': mlp, 'layer': layer, 'score': score, 'iters': mlp.n_iter_}


def test(x_train, x_test, y_train, y_test, path):
    if path and not os.path.exists(path):
        os.makedirs(path)

    scaler = StandardScaler()

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    in_size = len(x_train[0])
    ou_size = len(np.unique(y_train))

    nmin = min(int(in_size * 2 / 3) + ou_size, ou_size)

    configs = []
    for i in range(1, 5):
        for j in range(nmin // i, in_size // i):
            configs.append(train(tuple(j for _ in range(i)), x_train, x_test, y_train, y_test, save=path))
            print({k: configs[-1][k] for k in configs[-1] if k != 'mlp'})

    mx = max(x['score'] for x in configs)
    best = [_ for _ in configs if _['score'] == mx]
    print(f'Best performers: {best}')

    with open(f'{path}/summary.txt', 'w') as f:
        f.write(
            '\n'.join([str({k: x[k] for k in x if k != 'mlp'}) for x in configs]) +
            '\nBest performers: ' +
            '\n'.join([str({k: x[k] for k in x if k != 'mlp'}) for x in best]))
        f.close()
    for b in best:
        pickle.dump(b['mlp'], open(f'{path}/{b["layer"]}.p', 'wb'))
