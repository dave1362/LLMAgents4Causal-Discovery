<div align="center">

# (ACL 2025 Findings) Exploring Multi-Modal Integration with Tool-Augmented LLM Agents for Precise Causal Discovery

</div>

<p align="center">
    🔍&nbsp;<a href="#about">About</a>
    | 🚀&nbsp;<a href="#quick-start">Quick Start</a>
    | 📊&nbsp;<a heref="#results-and-evaluation"> Results and Evaluation</a>
    | 🔗&nbsp;<a href="#citation">Citation</a>
</p>


This is the official repository for paper "[Exploring Multi-Modal Integration with Tool-Augmented LLM Agents for Precise Causal Discovery](https://arxiv.org/abs/2412.13667)". We provide all the necessary code for both the main experiments and the ablation studies.



## 🔍About

MATMCD (Multi-Agent with Tool-Augmented LLMs for Multi-Modality Enhanced Causal Discovery) is a novel framework designed to improve the accuracy of causal discovery by integrating multi-modal data using large language model (LLM) agents.

### 🔧Framework

Traditional causal discovery methods rely solely on statistical correlations in observational data, overlooking valuable semantic cues from external sources. MATMCD supports modular integration with various causal discovery algorithms (e.g., PC, ES, DirectLiNGAM), and enables enhanced reasoning by combining symbolic graphs and unstructured textual data. MATMCD addresses this gap by introducing a multi-agent system consisting of:

<p align="center">
    <img src="./image/model_framework.jpg" alt="MATMCD" width="80%">
</p>

- Data Augmentation Agent (DA-AGENT): Retrieves and summarizes semantic context (e.g., from web or log data) using search tools and LLMs.
- Causal Constraint Agent (CC-AGENT): Integrates augmented data with initial causal graphs to verify or refute causal links using a reasoning pipeline.
- Causal Graph Refiner: Reconstructs the final causal graph by combining LLM-inferred constraints with traditional structure learning algorithms.

### 🔑Key Features

  - **Multi-source**: Integrates raw data, metadata, web documents, and logs to enrich semantic context for causal discovery. 
  - **LLM Reasoning**: Employs tool-augmented LLMs to reason over causal structures using external knowledge and contextual cues.
  - **Modular Design**: Features a modular architecture that allows easy swapping of LLMs and SCD algorithms for flexible adaptation.

## 🚀Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your_username/MATMCD.git
   cd MATMCD
   ```

2. **Set Up the Environment**
   - We recommend using `conda` or `virtualenv` to create an isolated environment.
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Configure API Keys**
   - add your API-keys in `config.py` file.

4. **Download the datasets**
    - Download the datasets have groundtruth from [AutoMPG](https://archive.ics.uci.edu/dataset/9/auto+mpg), [DWD Climate](https://webdav.tuebingen.mpg.de/cause-effect/), [Sachs](https://www.bnlearn.com/bnrepository/discrete-small.html#sachs), [Asic](https://www.bnlearn.com/bnrepository/discrete-small.html#asia), [Child](https://www.bnlearn.com/bnrepository/discrete-medium.html#child) and LEMMA_RCA datasets follow [LEMMA-RCA](https://lemma-rca.github.io/).
    - Create `data` folder and put the data inside of them.

5. **Run the Application**
    - Make sure your environment, API and dataset are accurate.
    - Feel free to run the `python GTdatasets_experiment.py` to start.

6. **Run Experiments and Evaluate**
    - Run benchmark experiments on standard datasets:
      ```bash
      python GTdatasets_experiment.py
      ```
    - For root cause analysis on microservice datasets:
      ```bash
      python RCA_experiment.py
      ```
    - Results will be saved in the `results/` folder.


## 📊 Results and Evaluation
  MATMCD is evaluated on:
  - **Benchmark Datasets**: AutoMPG, DWDClimate, SachsProtein, Asia, and Child — covering both time-series and discrete data.
  - **AIOps Datasets**: Product Review and Cloud Computing — large-scale multivariate time series with event logs.

  Key results:
  - **Up to 66.7% reduction in causal inference errors** (NHD) over SOTA methods.
  - **Up to 83.3% improvement in root cause ranking accuracy** (MAP@10).

<p align="center">
    <img src="./image/table1.png" alt="table1" width="80%">
</p>

<p align="center">
    <img src="./image/table2.png" alt="table2" width="80%">
</p>

## 🔗Citation

```
@inproceedings{shen2025MATMCD,
  title={Exploring Multi-Modal Integration with Tool-Augmented LLM Agents for Precise Causal Discovery},
  author={Shen, ChengAo and Chen, Zhengzhang and Luo, Dongsheng and Xu, Dongkuan and Chen, Haifeng and Ni, Jingchao},
  booktitle={ACL(Findings)},
  year={2025}
}
```

## ✨Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.

- https://github.com/mas-takayama/LLM-and-SCD
- https://github.com/superkaiba/causal-llm-bfs
- https://github.com/tavily-ai/tavily-python



## 📧Contract

If you have any questions or concerns, please contact us: cshen9@uh.edu or submit an issue

