import numpy as np
import pandas as pd
from Utils.metrics import Metrics
from Utils.RCA import random_walk_with_restart

datalist = ["Auto_MPG", "DWD_climate", "Sachs", "asia", "child"]

for name in datalist:
    print(f"current dataset: {name}")
    data = np.array(
        pd.read_csv(
            f"./causal-llm-bfs/causal_LLM_Matrix/{name}.csv", header=None
        ).values
    )
    GTmatrix = np.array(pd.read_csv(f"./data/{name}_GTmatrix.csv", header=None).values)

    print(len(GTmatrix[GTmatrix != 0]))
    print(len(data[data != 0]))

    print(data)
    print(GTmatrix)

    metrics = Metrics(data, GTmatrix)
    metrics.show_metrics()
datalist = ["Product_Review", "Cloud_Computing"]
for name in datalist:
    print(f"current dataset: {name}")
    adjacency_matrix = np.array(
        pd.read_csv(
            f"./causal-llm-bfs/causal_LLM_Matrix/{name}.csv", header=None
        ).values
    )
    labels = pd.read_csv(f"./data/LEMMA_RCA/{name}/Metrics/{name}.csv").columns.tolist()

    count = random_walk_with_restart(adjacency_matrix.T, len(labels) - 1)
    count_label_pairs = list(zip(count, labels))
    sorted_pairs = sorted(count_label_pairs, key=lambda x: x[0], reverse=True)
    print("\nRanked metrics by importance:")
    for count, label in sorted_pairs:
        print(f"{label}: {count:.4f}")
    print("================================================\n")
