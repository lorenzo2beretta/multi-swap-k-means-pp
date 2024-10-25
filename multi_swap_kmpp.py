import numpy as np
import random
import itertools
import copy
from tqdm import tqdm
import uuid
import time

# Compute the cost between two points as the squared Euclidean distance
def cost(a, b):
    if a is None or b is None:
        return np.inf
    diff = a - b
    return np.dot(diff, diff)


# Runs lloyd_iter many iterations of Lloyd's algorithm
def lloyd(dataset, centers, lloyd_iter):
    centers_series = [copy.copy(centers)]
    curr_ctrs = list(centers.values())
    cost_series = [sum(min(cost(p, cn) for cn in curr_ctrs) for p in dataset)]
    d = dataset[0].shape[0]
    for i in range(lloyd_iter):
        tmp_ctrs = [np.zeros(d) for _ in curr_ctrs]
        tmp_cluster_size = [0.0 for _ in curr_ctrs]
        for j, p in enumerate(dataset):
            idx = min((cost(p, cn), i) for i, cn in enumerate(curr_ctrs))[1]
            tmp_ctrs[idx] += p
            tmp_cluster_size[idx] += 1.0
        for j, _ in enumerate(curr_ctrs):
            if tmp_cluster_size[j] > 0.0:
                curr_ctrs[j] = tmp_ctrs[j] / tmp_cluster_size[j]
            else:
                curr_ctrs[j] = None
        cost_series.append(sum(min(cost(p, cn)
                                   for cn in curr_ctrs) for p in dataset))
        centers_series.append({uuid.uuid1(): ctr for ctr in curr_ctrs})
    return cost_series, centers_series


# Samples ssize points according to the D2-distribution, namely the
# distribution proportional to their current cost squared
def d2_sample(dataset, cost_table, ssize):
    tot_cost = sum(ls[0][0] for ls in cost_table)
    thresholds = sorted([random.uniform(0, tot_cost) for _ in range(ssize)])
    sample_idx = []
    for i, ls in enumerate(cost_table):
        if not thresholds:
            break
        tot_cost -= ls[0][0]
        while thresholds and thresholds[-1] >= tot_cost:
            sample_idx.append(i)
            thresholds.pop()
    return [(uuid.uuid1(), dataset[i]) for i in sample_idx]


# Finds a set of centers using the K-means++ algorithm
def kmpp(dataset, k):
    first_center = dataset[random.randint(0, len(dataset))]
    first_id = uuid.uuid1()
    centers = {first_id: first_center}
    cost_table = [[(cost(x, first_center), first_id)] for x in dataset]
    for j in range(1, k):
        ctr_id, ctr = d2_sample(dataset, cost_table, 1)[0]
        centers[ctr_id] = ctr
        for i, x in enumerate(dataset):
            cost_table[i].append((cost(x, ctr), ctr_id))
            cost_table[i].sort()
    return cost_table, centers

# Greedily selects the center to swap out
def greedy_out(cost_table, centers, k):
    curr_cost = sum(ls[0][0] for ls in cost_table)
    out = []
    curr_centers = copy.copy(centers)
    while len(curr_centers) > k:
        cost_jump = {key: 0.0 for key in curr_centers}
        for ls in cost_table:
            i = 0
            while ls[i][1] in out:
                i += 1
            j = i + 1
            while ls[j][1] in out:
                j += 1
            cost_jump[ls[i][1]] += ls[j][0] - ls[i][0]

        last_out = min((cost_jump[key], key) for key in curr_centers)[1]
        curr_cost += cost_jump[last_out]
        out.append(last_out)
        del curr_centers[last_out]
    return curr_cost, curr_centers


# Implementation of a single step of multi-swap K-means++ algorithm with greedy selection process
def greedy_msls_step(dataset, cost_table, centers, k, swap_size):
    samples = d2_sample(dataset, cost_table, swap_size)
    curr_centers = copy.copy(centers)
    for ctr_id, ctr in samples:
        curr_centers[ctr_id] = ctr
    old_cost = sum(ls[0][0] for ls in cost_table)
    for i, x in enumerate(dataset):
        for ctr_id, ctr in samples:
            cost_table[i].append((cost(ctr, x), ctr_id))
        cost_table[i].sort()

    greedy_cost, greedy_centers = greedy_out(cost_table, curr_centers, k)
    if greedy_cost < old_cost:
        centers = greedy_centers
    for i, _ in enumerate(cost_table):
        cost_table[i] = [(cst, ctr) for cst, ctr in cost_table[i]
                         if ctr in centers]
    return centers

# Implementation of the multi-swap K-means++ algorithm with greedy selection process
def greedy_msls(dataset, cost_table, centers, k, swap_size, ls_iter,
                time_limit=None):
    cost_series = [sum([ls[0][0] for ls in cost_table])]
    center_series = [centers]
    if not time_limit:
        for _ in range(ls_iter):
            centers = greedy_msls_step(
                dataset, cost_table, centers, k, swap_size)
            center_series.append(centers)
            cost_series.append(sum(ls[0][0] for ls in cost_table))
    else:
        toc = tic = time.process_time()
        while toc - tic < time_limit:
            centers = greedy_msls_step(
                dataset, cost_table, centers, k, swap_size)
            center_series.append(centers)
            cost_series.append(sum(ls[0][0] for ls in cost_table))
            toc = time.process_time()
    return cost_series, center_series


# Implementation of  a single step of the multi-swap K-means++ algorithm 
# with brute-force (or vanilla) selection process
def vanilla_msls_step(dataset, cost_table, centers, k, swap_size):
    samples = d2_sample(cost_table, swap_size)
    all_centers = centers + samples
    best_cost = sum(ls[0][0] for ls in cost_table)
    old_cost = best_cost
    best_out = samples
    for i, x in enumerate(dataset):
        for q in samples:
            cost_table[i].append((cost(dataset[q], x), q))
        cost_table[i].sort()

    random.shuffle(all_centers)
    for out in itertools.combinations(all_centers, swap_size - 1):
        curr_cost = 0
        curr_centers = copy.copy(all_centers)
        for x in out:
            curr_centers.remove(x)
        cost_jump = {cn: 0.0 for cn in curr_centers}
        for ls in cost_table:
            i = 0
            while ls[i][1] in out:
                i += 1
            curr_cost += ls[i][0]
            j = i + 1
            while ls[j][1] in out:
                j += 1
            cost_jump[ls[i][1]] += ls[j][0] - ls[i][0]

        last_out = min((cost_jump[cn], cn) for cn in curr_centers)[1]
        curr_out = list(out) + [last_out]
        curr_cost += cost_jump[last_out]
        curr_centers.remove(last_out)

        if curr_cost < best_cost:
            best_cost = curr_cost
            best_out = curr_out
            if best_cost < (1 - 1/(4 * k)) * old_cost:
                break

    final_centers = [c for c in all_centers if c not in best_out]
    for i, _ in enumerate(cost_table):
        cost_table[i] = [(cst, ctr) for cst, ctr in cost_table[i]
                         if ctr in final_centers]
    return final_centers


# Implementation of the multi-swap K-means++ algorithm with brute-force (or vanilla) 
# selection process
def vanilla_msls(dataset, cost_table, centers, k, swap_size, ls_iter):
    cost_series = [sum(ls[0][0] for ls in cost_table)]
    center_series = [centers]
    for _ in tqdm(range(ls_iter)):
        centers = vanilla_msls_step(dataset, cost_table, centers, k, swap_size)
        center_series.append(centers)
        cost_series.append(sum(ls[0][0] for ls in cost_table))
    return cost_series, center_series
