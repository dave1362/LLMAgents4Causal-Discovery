from ConstrainAgent.ConstrainAgent import OnlyReactAgent, OnlyLLMAgent
from Utils.CausalDiscovery import (
    causal_discovery,
)
from Utils.data import load_Lemma_data
from Utils.visualize import visualize_graph
from Utils.RCA import random_walk_with_restart

causal_discovery_algorithm = "pc"

system_name = "Product_Review"
theme = "MicroService system about Product Review"
day = 20210517

# system_name = "Cloud_Computing"
# theme = "MicroService system about Cloud Computing"
# day = 20231207


print(f"Loading dataset: {system_name}_{day}...")
data_table, log_dir = load_Lemma_data(system_name, day)
data = data_table.values
labels = data_table.columns.tolist()

print("================================================\n")

print("Running PC algorithm...")
adjacency_matrix = causal_discovery(data, labels, method=causal_discovery_algorithm)
visualize_graph(adjacency_matrix, labels, f"./image/{system_name}_{day}/PC_graph.png")
count = random_walk_with_restart(adjacency_matrix, len(labels) - 1)
count_label_pairs = list(zip(count, labels))
sorted_pairs = sorted(count_label_pairs, key=lambda x: x[0], reverse=True)
print("\nRanked metrics by importance:")
for count, label in sorted_pairs:
    print(f"{label}: {count:.4f}")
print("================================================\n")

print("Running Only LLM Agent...")
constrain_agent = OnlyLLMAgent(
    labels,
    theme,
    graph_matrix=adjacency_matrix,
    causal_discovery_algorithm=causal_discovery_algorithm,
)

constraint_matrix = constrain_agent.run()

adjacency_matrix_optimized = causal_discovery(
    data,
    labels,
    method=causal_discovery_algorithm,
    constraint_matrix=constraint_matrix,
)

visualize_graph(adjacency_matrix, labels, f"./image/{system_name}_{day}/PC_graph.png")
count = random_walk_with_restart(adjacency_matrix, len(labels) - 1)
count_label_pairs = list(zip(count, labels))
sorted_pairs = sorted(count_label_pairs, key=lambda x: x[0], reverse=True)
print("\nRanked metrics by importance:")
for count, label in sorted_pairs:
    print(f"{label}: {count:.4f}")
print("================================================\n")

print("Running Only React Agent...")
constrain_agent = OnlyReactAgent(
    labels,
    theme,
    graph_matrix=adjacency_matrix,
    causal_discovery_algorithm=causal_discovery_algorithm,
)

constraint_matrix = constrain_agent.run()

adjacency_matrix_optimized = causal_discovery(
    data,
    labels,
    method=causal_discovery_algorithm,
    constraint_matrix=constraint_matrix,
)
visualize_graph(adjacency_matrix, labels, f"./image/{system_name}_{day}/PC_graph.png")
count = random_walk_with_restart(adjacency_matrix, len(labels) - 1)
count_label_pairs = list(zip(count, labels))
sorted_pairs = sorted(count_label_pairs, key=lambda x: x[0], reverse=True)
print("\nRanked metrics by importance:")
for count, label in sorted_pairs:
    print(f"{label}: {count:.4f}")
print("================================================\n")
