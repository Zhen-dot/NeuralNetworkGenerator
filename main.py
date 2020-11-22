from functions import *
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

tra, tes = load_digits()
test(tra['data'], tes['data'], tra['target'], tes['target'], 'results')
