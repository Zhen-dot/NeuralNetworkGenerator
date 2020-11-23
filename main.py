from functions import *
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    # Load training & testing data
    tra, tes = load_digits()

    # Loads and describes the best classifier
    mlp = load_mlp('results/(29,).p', tra, tes)

    # Tests various topologies to find best topology
    # test(tra['data'], tes['data'], tra['target'], tes['target'], 'results')
