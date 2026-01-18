import pandas as pd
import numpy as np
import arff

from sklearn.metrics import accuracy_score
from vowpalwabbit import pyvw
from flaml import AutoVW
from river import metrics
import string
import sys

from collecter import WindowClassificationPerformanceEvaluator
from codecarbon import OfflineEmissionsTracker

import psutil
import time

import json

import argparse

import os 
import sys
# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser(description="Run ChaCha AutoVW Stream Learner")

parser.add_argument(
    "dataset_name",
    type=str,
    help="Name of the dataset (e.g., forest_cover)"
)

parser.add_argument(
    "--ensemble_size",
    type=int,
    default=5,
    help="Number of live models in AutoVW (default: 5)"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed (default: 42)"
)

args = parser.parse_args()

# Apply arguments
dataset_name = args.dataset_name
ensemble_size = args.ensemble_size
random_seed = args.seed


# Retrieve and prepare data
B = pd.read_csv(f"stream_datasets/{dataset_name}.csv")


# Preprocessing of data: Drop NaNs, move target to the end, check for zero values
x = B.drop("class", axis=1)
y = B["class"]

if pd.isnull(B.iloc[:, :]).any().any():
    print(
        "Data X contains NaN values. The rows that contain NaN values will be dropped."
    )
    # B.dropna(inplace=True)

if B[:].iloc[:, 0:-1].eq(0).any().any():
    print(
        "Data contains zero values. They are not removed but might cause issues with some River learners."
    )

X = B[:].iloc[:, 0:-1]
y = B[:].iloc[:, -1]

#################################
# Convert into vowpalwabbit examples:
NS_LIST = list(string.ascii_lowercase) + list(string.ascii_uppercase)
max_ns_num = 10  # the maximum number of namespaces
orginal_dim = X.shape[1]
max_size_per_group = int(np.ceil(orginal_dim / float(max_ns_num)))
# sequential grouping

group_indexes = []
for i in range(max_ns_num):
    indexes = [
        ind
        for ind in range(
            i * max_size_per_group, min((i + 1) * max_size_per_group, orginal_dim)
        )
    ]
    if len(indexes) > 0:
        group_indexes.append(indexes)

vw_examples = []
for i in range(X.shape[0]):

    ns_content = []
    for zz in range(len(group_indexes)):
        ns_features = " ".join(
            "{}:{:.6f}".format(ind, X.iloc[i, ind]) for ind in group_indexes[zz]
        )
        ns_content.append(ns_features)
    ns_line = "{} |{}".format(
        str(y[i]),
        "|".join(
            "{} {}".format(NS_LIST[j], ns_content[j]) for j in range(len(group_indexes))
        ),
    )
    vw_examples.append(ns_line)

###################################

max_iter_num = len(vw_examples)

wcpe = WindowClassificationPerformanceEvaluator(metric=metrics.Accuracy(),
                                                    window_width=1000,
                                                    print_every=1000)

scores = []
times = []
memories = []
emissions = []
energy = []
tracker=OfflineEmissionsTracker(country_iso_code="EST",log_level="critical",allow_multiple_runs=True)
tracker.start()

online_metric = metrics.Accuracy()

# setup autoVW
autovw_ni = AutoVW(
    max_live_model_num=ensemble_size,
    search_space={"interactions": AutoVW.AUTOMATIC, "quiet": ""},
    random_seed=random_seed
)



# online learning
for i in range(max_iter_num):
    mem_before = psutil.Process(os.getpid()).memory_info().rss  # Recording Memory
    tracker.start_task()
    start = time.time() 

    vw_x = vw_examples[i]
    y_true = float(vw_examples[i].split("|")[0])
    # predict step
    y_pred = autovw_ni.predict(vw_x)
    # update online metric
    online_metric = online_metric.update(int(y_true), round(y_pred))
    # learn step

    autovw_ni.learn(vw_x)
    
    end = time.time()


    iteration_time = end - start
    mem_after = psutil.Process(os.getpid()).memory_info().rss
    iteration_mem = mem_after - mem_before
    
    memories.append(iteration_mem)
    emission_record=tracker.stop_task()
    scores.append(online_metric)
    times.append(abs(iteration_time))
    emissions.append(emission_record.emissions)
    energy.append(emission_record.energy_consumed)



save_record = {
        "model": "ChaCha",
        "dataset": dataset_name,
        "prequential_scores": scores,
        "windows_scores": [],
        "time": times,
        "memory": memories,
        "emission": emissions,
        "energy_consumed": energy #kwh
    }

dir_path = "experiment-results/ChaCha"
os.makedirs(dir_path, exist_ok=True)  # Make sure the directory exists


file_name = f"{save_record['model']}_{save_record['dataset']}_seed_{random_seed}_ensemble_size_{ensemble_size}.json"  
file_path = os.path.join(dir_path, file_name)