import numpy as np

# This code is for v1 of the openai package: pypi.org/project/openai
from openai import OpenAI
from .bfs_prompts import *

client = OpenAI(api_key="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")


model = "gpt-4o-mini"
temperature = 0.7


def llm_bfs(var_names_and_desc, dataset, df, include_statistics=False):
    if include_statistics:
        corr = df.corr()

    if dataset == "asia":
        message_history = asia_prompt
    elif dataset == "child":
        message_history = child_prompt
    elif dataset == "Auto_MPG":
        message_history = auto_mpg_prompt
    elif dataset == "DWD_climate":
        message_history = dwd_climate_prompt
    elif dataset == "Sachs":
        message_history = sachs_prompt
    elif dataset == "Product_Review":
        message_history = product_review_prompt
    elif dataset == "Cloud_Computing":
        message_history = cloud_computing_prompt
    nodes = [var for var in var_names_and_desc]
    for var in var_names_and_desc:
        causal_var = var_names_and_desc[var]
        message_history[1]["content"] += f"""{var}: {causal_var.description}\n"""

    message_history[1]["content"] += prompt_init
    print(message_history[1]["content"])

    response = client.chat.completions.create(
        model=model,
        messages=message_history,
        temperature=temperature,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    answer = response.choices[0].message.content
    print(answer)
    message_history.append({"role": "assistant", "content": answer})
    answer = answer.split("<Answer>")[1].split("</Answer>")[0].split(", ")
    print(answer)

    independent_nodes = answer.copy()
    unvisited_nodes = nodes.copy()
    print(f"unvisited_nodes: {unvisited_nodes}")
    print(f"independent_nodes: {independent_nodes}")
    print(f"answer: {answer}")

    for node in answer:
        if node == "":
            continue
        unvisited_nodes.remove(node)
    frontier = []

    predict_graph = dict()

    for to_visit in independent_nodes:
        prompt = (
            "Given "
            + ", ".join(independent_nodes)
            + " is(are) not affected by any other variable"
        )
        if len(predict_graph) == 0:
            prompt += ".\n"
        else:
            prompt += " and the following causal relationships.\n"
            for head, tails in predict_graph.items():
                if len(tails) > 0:
                    prompt += f"{head} causes " + ", ".join(tails) + "\n"

        prompt += f"Select variables that are caused by {to_visit}.\n"

        if include_statistics:
            prompt += get_data_prompt(to_visit, corr)

        prompt += prompt_format
        print(prompt)

        message_history.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=message_history,
            temperature=temperature,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        answer = response.choices[0].message.content
        message_history.append({"role": "assistant", "content": answer})
        answer = answer.split("<Answer>")[1].split("</Answer>")[0].split(", ")
        print(answer)
        for node in answer:
            if node in independent_nodes:
                answer.remove(node)
            if len(node) == 0:
                answer.remove(node)
            elif node not in nodes:
                print("ERROR: ", node)
                answer.remove(node)

        predict_graph[to_visit] = answer
        for node in answer:
            if node in unvisited_nodes and node not in frontier:
                frontier.append(node)

    while len(frontier) > 0:
        print("Frontier: ", frontier)
        print("Unvisited nodes: ", unvisited_nodes)
        to_visit = frontier.pop(0)
        unvisited_nodes.remove(to_visit)
        print("Visiting: ", to_visit)
        prompt = (
            "Given "
            + ", ".join(independent_nodes)
            + " is(are) not affected by any other variable and the following causal relationships.\n"
        )
        for head, tails in predict_graph.items():
            if len(tails) > 0:
                prompt += f"{head} causes " + ", ".join(tails) + "\n"

        prompt += f"Select variables that are caused by {to_visit}.\n"

        if include_statistics:
            prompt += get_data_prompt(to_visit, corr)

        prompt += prompt_format

        print(prompt)
        print("Start generating...")
        message_history.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model,
            messages=message_history,
            temperature=temperature,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        answer = response.choices[0].message.content
        message_history.append({"role": "assistant", "content": answer})
        answer = answer.split("<Answer>")[1].split("</Answer>")[0].split(", ")
        print(answer)

        for node in answer:
            if node in independent_nodes:
                answer.remove(node)
            if len(node) == 0:
                answer.remove(node)
            elif node not in nodes:
                print("ERROR: ", node)
                answer.remove(node)

        predict_graph[to_visit] = answer
        for node in answer:
            if node in unvisited_nodes and node not in frontier:
                frontier.append(node)

    print(predict_graph)
    df_order = [var for var in df.columns]
    n = len(df_order)
    adj_matrix = np.zeros((n, n))
    for i, var in enumerate(df_order):
        if var in predict_graph:
            for node in predict_graph[var]:
                j = df_order.index(node)
                adj_matrix[i][j] = 1
    print(adj_matrix)
    return adj_matrix
