from ConstrainAgent.ConstrainAgent import ConstrainReactAgent
from Utils.CausalDiscovery import (
    causal_discovery,
)
from Utils.data import load_data_from_csv
from Utils.metrics import Metrics
from Utils.visualize import visualize_graph

data_theme = {
    "Auto_MPG": "Gasoline consumption",
    "DWD_climate": "Climate change",
    "Sachs": "Biology",
    "asia": "Lung Cancer",
    "child": "Infant Health Status",
}

causal_discovery_algorithm = "pc"


def experiment(dataset, theme):
    print(f"Loading dataset: {dataset}...")
    data, GTmatrix, labels = load_data_from_csv(dataset)
    visualize_graph(GTmatrix, labels, f"./image/{dataset}/GT_graph.png")
    print("================================================\n")

    print("Running PC algorithm...")
    adjacency_matrix = causal_discovery(data, labels, method=causal_discovery_algorithm)
    visualize_graph(adjacency_matrix, labels, f"./image/{dataset}/PC_graph.png")
    Metrics(adjacency_matrix, GTmatrix).show_metrics()
    print("================================================\n")

    print("Running ConstrainAgent...")
    constrain_agent = ConstrainReactAgent(
        labels,
        theme,
        graph_matrix=adjacency_matrix,
        causal_discovery_algorithm=causal_discovery_algorithm,
    )

    constraint_matrix = constrain_agent.run(use_cache=False)

    adjacency_matrix_optimized = causal_discovery(
        data,
        labels,
        method=causal_discovery_algorithm,
        constraint_matrix=constraint_matrix,
    )
    visualize_graph(
        adjacency_matrix_optimized, labels, f"./image/{dataset}/PC_graph_Optimized.png"
    )
    Metrics(adjacency_matrix_optimized, GTmatrix).show_metrics()
    print("================================================\n")


for dataset, theme in data_theme.items():
    experiment(dataset, theme)
