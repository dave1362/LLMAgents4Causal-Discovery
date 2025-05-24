from ConstrainAgent.ConstrainAgent import ConstrainNormalAgent
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

    print("Running ConstrainAgent...")
    constrain_agent = ConstrainNormalAgent(
        labels,
        theme,
        graph_matrix=adjacency_matrix,
        causal_discovery_algorithm=causal_discovery_algorithm,
        use_reasoning=False,
    )

    constraint_matrix = constrain_agent.run(
        use_cache=False,
        cache_path=f"./cache/Domain_knowledge/{dataset}/{causal_discovery_algorithm}",
    )

    print("The constraint matrix is:")
    print(constraint_matrix)

    adjacency_matrix_optimized = causal_discovery(
        data,
        labels,
        method=causal_discovery_algorithm,
        constraint_matrix=constraint_matrix,
    )
    print("The optimized adjacency matrix is:")
    print(adjacency_matrix_optimized)

    visualize_graph(
        adjacency_matrix_optimized,
        labels,
        f"./image/{dataset}/{causal_discovery_algorithm}_CCAgent.png",
    )
    Metrics(adjacency_matrix_optimized, GTmatrix).show_metrics()
    print("================================================\n")

    print("Running ConstrainAgent with Information Provided without Reasoning...")
    # data_info, node_info = request_web_information(dataset, labels, "default") # Web Fix search

    data_info, node_info = split_summary_into_sub_questions(
        open(f"./cache/Summarized_info/{dataset}_info.txt").read()
    )

    print(data_info)
    print(node_info)

    constrain_agent = ConstrainNormalAgent(
        labels,
        theme,
        graph_matrix=adjacency_matrix,
        causal_discovery_algorithm=causal_discovery_algorithm,
        dataset_information=data_info,
        node_information=node_info,
        use_reasoning=False,
    )

    prompt, system_prompt = constrain_agent.domain_knowledge_LLM.generate_prompt(
        0, 1, node_info
    )

    constraint_matrix = constrain_agent.run(
        use_cache=True,
        cache_path=f"./cache/Domain_knowledge/{dataset}/{causal_discovery_algorithm}",
    )

    print("The constraint matrix is:")

    adjacency_matrix_optimized = causal_discovery(
        data,
        labels,
        method=causal_discovery_algorithm,
        constraint_matrix=constraint_matrix,
    )

    print("The optimized adjacency matrix is:")
    print(adjacency_matrix_optimized)

    visualize_graph(
        adjacency_matrix_optimized,
        labels,
        f"./image/{dataset}/{causal_discovery_algorithm}_MATMCD.png",
    )
    Metrics(adjacency_matrix_optimized, GTmatrix).show_metrics()
    print("================================================\n")

    print("Running ConstrainAgent with Information Provided with Reasoning...")

    data_info, node_info = split_summary_into_sub_questions(
        open(f"./cache/Summarized_info/{dataset}_info.txt").read()
    )

    print(data_info)
    print(node_info)

    constrain_agent = ConstrainNormalAgent(
        labels,
        theme,
        graph_matrix=adjacency_matrix,
        causal_discovery_algorithm=causal_discovery_algorithm,
        dataset_information=data_info,
        node_information=node_info,
        use_reasoning=True,
    )

    prompt, system_prompt = constrain_agent.domain_knowledge_LLM.generate_prompt(
        0, 1, node_info
    )

    constraint_matrix = constrain_agent.run(
        use_cache=True,
        cache_path=f"./cache/Domain_knowledge/{dataset}/{causal_discovery_algorithm}",
    )

    print("The constraint matrix is:")
    print(constraint_matrix)

    adjacency_matrix_optimized = causal_discovery(
        data,
        labels,
        method=causal_discovery_algorithm,
        constraint_matrix=constraint_matrix,
    )

    print("The optimized adjacency matrix is:")
    print(adjacency_matrix_optimized)

    visualize_graph(
        adjacency_matrix_optimized,
        labels,
        f"./image/{dataset}/{causal_discovery_algorithm}_MATMCD-R.png",
    )
    Metrics(adjacency_matrix_optimized, GTmatrix).show_metrics()
    print("================================================\n")


for dataset, theme in data_theme.items():
    experiment(dataset, theme)
