from ConstrainAgent.ConstrainAgent import ConstrainNormalAgent, OnlyLLMAgent
from Utils.CausalDiscovery import (
    causal_discovery,
)
from Utils.data import load_data_from_csv
from Utils.metrics import Metrics
from Utils.visualize import visualize_graph
from Web_tools import split_summary_into_sub_questions

data_theme = {
    "Auto_MPG": "Gasoline consumption",
    "DWD_climate": "Climate change",
    "Sachs": "Biology",
    "asia": "Lung Cancer",
    "child": "Infant Health Status",
}

causal_discovery_algorithm = "pc"
# causal_discovery_algorithm = "Exact-Search"
# causal_discovery_algorithm = "DirectLiNGAM"


def experiment(dataset, theme):
    print(f"Loading dataset: {dataset}...")
    data, GTmatrix, labels = load_data_from_csv(dataset)
    visualize_graph(GTmatrix, labels, f"./image/{dataset}/GT_graph.png")
    print("================================================\n")

    print(f"Running {causal_discovery_algorithm} algorithm...")
    adjacency_matrix = causal_discovery(data, labels, method=causal_discovery_algorithm)
    visualize_graph(
        adjacency_matrix,
        labels,
        f"./image/{dataset}/{causal_discovery_algorithm}_graph.png",
    )
    Metrics(adjacency_matrix, GTmatrix).show_metrics()
    print("================================================\n")

    data_info, node_info = split_summary_into_sub_questions(
        open(f"./cache/Summarized_info/{dataset}_info.txt").read()
    )

    print(data_info)
    print(node_info)

    constrain_agent = OnlyLLMAgent(
        labels,
        theme,
        graph_matrix=adjacency_matrix,
        causal_discovery_algorithm=causal_discovery_algorithm,
        dataset_information=data_info,
        node_information=node_info,
        use_reasoning=True,
        guess_number=2,
    )

    adjacency_matrix_optimized = causal_discovery(
        data,
        labels,
        method=causal_discovery_algorithm,
        constraint_matrix=constrain_agent.run(),
    )

    print("The optimized adjacency matrix is:")
    print(adjacency_matrix_optimized)

    # visualize_graph(
    #     adjacency_matrix_optimized,
    #     labels,
    #     f"./image/{dataset}/{causal_discovery_algorithm}_MATMCD.png",
    # )
    Metrics(adjacency_matrix_optimized, GTmatrix).show_metrics()
    print("================================================\n")


for dataset, theme in data_theme.items():
    experiment(dataset, theme)
