asia_prompt = [
    {"role": "system", "content": "You are an expert on lung diseases."},
    {
        "role": "user",
        "content": "You are a helpful assistant to experts in lung diesease research. Our goal is to construct a causal graph between the following variables.\n",
    },
]


child_prompt = [
    {"role": "system", "content": "You are an expert on children's diseases."},
    {
        "role": "user",
        "content": "You are a helpful assistant to experts in children's diesease research. Our goal is to construct a causal graph between the following variables.\n",
    },
]

auto_mpg_prompt = [
    {"role": "system", "content": "You are an expert on cars."},
    {
        "role": "user",
        "content": "You are a helpful assistant to experts in car research. Our goal is to construct a causal graph between the following variables.\n",
    },
]

dwd_climate_prompt = [
    {"role": "system", "content": "You are an expert on climate."},
    {
        "role": "user",
        "content": "You are a helpful assistant to experts in climate research. Our goal is to construct a causal graph between the following variables.\n",
    },
]

sachs_prompt = [
    {"role": "system", "content": "You are an expert on molecular biology."},
    {
        "role": "user",
        "content": "You are a helpful assistant to experts in molecular biology research. Our goal is to construct a causal graph between the following variables.\n",
    },
]

product_review_prompt = [
    {"role": "system", "content": "You are an expert on product reviews."},
    {
        "role": "user",
        "content": "You are a helpful assistant to experts in product review research. Our goal is to construct a causal graph between the following variables.\n",
    },
]

cloud_computing_prompt = [
    {"role": "system", "content": "You are an expert on cloud computing."},
    {
        "role": "user",
        "content": "You are a helpful assistant to experts in cloud computing research. Our goal is to construct a causal graph between the following variables.\n",
    },
]

prompt_init = "Now you are going to use the data to construct a causal graph. You will start with identifying the variable(s) that are unaffected by any other variables.\n"
prompt_format = 'Think step by step. Then, provide your final answer (variable names only) within the tags <Answer>...</Answer>, seperated by ", ". '
prompt_init += prompt_format


def get_data_prompt(to_visit, corr):
    prompt = f"Addtionally, the Pearson correlation coefficient between {to_visit} and other variables are as follows:\n"
    for var in corr:
        if var == to_visit:
            continue
        prompt += f"{var}: {corr[var][to_visit]:.2f}\n"
    return prompt
