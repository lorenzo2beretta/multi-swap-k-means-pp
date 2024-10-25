from matplotlib import pyplot as plt
import pandas as pd
import copy
import time
from tqdm import tqdm
import numpy as np
import statistics
import sys
from multi_swap_kmpp import greedy_msls
from multi_swap_kmpp import vanilla_msls
from multi_swap_kmpp import kmpp
from multi_swap_kmpp import lloyd
from utilities import read

dataset_name = sys.argv[1]
k = int(sys.argv[2])
ls_iter = int(sys.argv[3])
lloyd_iter = int(sys.argv[4])
num_experiments = int(sys.argv[5])
out_dir_name = sys.argv[6]

dataset = read("datasets/" + dataset_name)
ssizes = [1, 4, 7, 10]

cost_df = pd.DataFrame()
time_df = pd.DataFrame()

for exp in tqdm(range(num_experiments)):
    original_ct, original_ctrs = kmpp(dataset, k)

    cost_series = lloyd(dataset, original_ctrs, lloyd_iter)[0]
    for i, x in enumerate(cost_series):
        cost_df = pd.concat([cost_df, pd.DataFrame({'swap_size': ['KM++'],
                                                    'experiment': [exp],
                                                    'iteration': [i],
                                                    'cost': [x]})])
    for ss in ssizes:
        print("---------------SWAP SIZE " + str(ss) + "---------------")
        cost_table = copy.deepcopy(original_ct)
        centers = copy.deepcopy(original_ctrs)

        ls_time = -time.process_time()
        cost_series, centers_series = greedy_msls(
            dataset, cost_table, centers, k, ss, ls_iter)
        ls_time += time.process_time()

        lloyd_time = -time.process_time()
        cost_series_ll, centers_series_ll = lloyd(
            dataset, centers_series[-1], lloyd_iter)
        lloyd_time += time.process_time()

        time_df = pd.concat([time_df, pd.DataFrame({'swap_size': [ss],
                                                    'experiment': [exp],
                                                    'seeding': [ls_time],
                                                    'lloyd': [lloyd_time]})])

        for i, x in enumerate(cost_series_ll):
            cost_df = pd.concat([cost_df, pd.DataFrame({'swap_size': [ss],
                                                        'experiment': [exp],
                                                        'iteration': [i],
                                                        'cost': [x]})])

# Write results on file
outfile_name = dataset_name + f"-k={k}-ls_iter={ls_iter}"
cost_df.to_csv("../out_dir_name/" +
               outfile_name, index=False)
time_df.to_csv("../out_dir_name/" +
               outfile_name, index=False)
