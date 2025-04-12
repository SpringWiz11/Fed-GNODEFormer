# Federated Spectral Graph Transformers Meet Neural Ordinary Differential Equations for Non-IID Graphs

Accepted at **Transactions on Machine Learning Research (TMLR) 2025**

## Abstract

> Graph Neural Network (GNN) research is rapidly advancing due to GNNs’ capacity to learn
distributed representations from graph-structured data. However, centralizing large volumes
of real-world graph data for GNN training is often impractical due to privacy concerns, regulatory restrictions, and commercial competition. Federated learning (FL), a distributed
learning paradigm, offers a solution by preserving data privacy with collaborative model
training. Despite progress in training huge vision and language models, federated learning
for GNNs remains underexplored. To address this challenge, we present a novel method for
federated learning on GNNs based on spectral GNNs equipped with neural ordinary differential equations (ODE) for better information capture, showing promising results across
both homophilic and heterophilic graphs. Our approach effectively handles non-Independent
and Identically Distributed (non-IID) data, while also achieving performance comparable
to existing methods that only operate on IID data. It is designed to be privacy-preserving
and bandwidth-optimized, making it suitable for real-world applications such as social network analysis, recommendation systems, and fraud detection, which often involve complex,
non-IID, and heterophilic graph structures. Our results in the area of federated learning
on non-IID heterophilic graphs demonstrate significant improvements, while also achieving
better performance on homophilic graphs. This work highlights the potential of federated
learning in diverse and challenging graph settings.
## Paper

Link to the paper: [https://openreview.net/](https://openreview.net/forum?id=TR6iUG8i6Z)  
<!-- Replace with your actual OpenReview link -->

## Getting Started


We use **[Poetry](https://python-poetry.org/)** for dependency and environment management.

## Installation

### 1. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Clone the repository

```bash
https://github.com/SpringWiz11/Fed-GNODEFormer.git
cd Fed-GNODEFormer
```

### 3. Install dependencies using Poetry

```bash
poetry install
```

### 4. Activate the Poetry environment

```bash
poetry shell
```

## Running Experiments

We support two execution modes:

### ➤ Federated Mode

Run the experiment in a **federated learning** setup:

```bash
python3 runner.py -m --mode federated
```

### ➤ Centralized Mode

Run the experiment in a **centralized learning** setup:

```bash
python3 runner.py -m --mode centralized
```



## 🤝 Contributing

We welcome contributions, issues, and discussions!  
Please open an issue or submit a pull request if you have ideas to improve the project.
