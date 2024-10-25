import numpy as np
import pandas as pd

# Reads program arguments and reads dataset from file and returns
def read(filename):
    file = open(filename, "r")
    dataset = [np.array([float(c) for c in line.split(",")]) for line in file]
    return dataset

# this function cleans and rescale the dataset so that all numbers are in [0, 1]
def clean_and_rescale(filename, suffix='-scaled'):
    file = open("datasets/" + filename, "r")
    tmp = [[c for c in line.split(",")] for line in file]
    for i, _ in enumerate(tmp):
        tmp[i][0] = float(tmp[i][0].split(" ")[1])
        for j in range(1, len(tmp[i]) - 1):
            tmp[i][j] = float(tmp[i][j].split(" ")[2])
        tmp[i][-1] = float(tmp[i][-1].split(" ")[2].split("}")[0])
    ds = pd.DataFrame(tmp)
    ds_clean = ds[~ds.isna().any(axis=1)]
    ds_scaled = (ds_clean - ds_clean.min()) / (ds_clean.max() - ds_clean.min())
    ds_scaled.to_csv('datasets/' + filename.split('.')[0] + suffix + '.txt', index=False, header=False)
