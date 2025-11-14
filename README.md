# 🚀 GraphSAGE+GAT-based Influencer Intelligence System Using Graph Neural Networks

Unlock hidden influencers, uncover communities, and analyze social dynamics—using by **GraphSAGE, GAT,** and **advanced graph analytics**.
This project transforms raw interaction data into a **high-resolution social influence map**, combining **GNN-based embeddings**, **network centrality**, and **dynamic visualization** into a single interactive intelligence dashboard.

## 📌 Introduction

Social networks are not just connections—they are complex, evolving ecosystems.
Finding influential individuals in such environments requires understanding:
- Who talks to whom?
- How information flows?
- Which nodes act as bridges, hubs, or bottlenecks?
- Which users have structural or hidden influence?

This project tackles that challenge using Graph Neural Networks (GNNs) and classical Network Science.
We build a **hybrid influence scoring pipeline** where:
- **GNNs (GraphSAGE + GAT)** learn deep relational embeddings
- Centrality metrics (PageRank, Betweenness) quantify global & local influence
- **t-SNE** projections reveal community clusters
- A **Streamlit dashboard** brings everything together for explorable, human-friendly analytics

The final system not only ranks influencers—it explains why they matter, how they are positioned in the network, and what structural role they play.

## Models used
### 🔹 GraphSAGE
- Learns node embeddings through neighborhood sampling & aggregation.
- Scales efficiently to large networks.
- Captures generalized interaction patterns.

### 🔹 Graph Attention Networks (GAT)
- Uses attention coefficients to learn importance of each neighbor.
- Helps identify influential interactions automatically.
- Great for heterogeneous or noisy graphs.

Both models are trained on the processed interaction graph and later used to generate high-quality node embeddings for influence scoring.

## ⚙️ Methodolgy
### Hybrid Ranking Score
We normalize and combine:
```bash
combined_score = norm(emb_norm) + norm(pagerank) + norm(betweenness)
```
This blends:
- Data-driven learning (GNN embeddings)
- Graph-theoretic importance (centrality)

Note: Every feature, model, and visualization in this repository is powered exclusively by the Reddit dataset.

### Visualization using t-SNE
We create a 2D map of the embedding space:
```bash
tsne = TSNE(n_components=2)
emb_2d = tsne.fit_transform(embeddings)
```
where Each dot represents a user.
Nodes closer together → more similar influence patterns.

### 🎛️ Streamlit Dashboard (Interactive Analytics)
<img width="1919" height="910" alt="image" src="https://github.com/user-attachments/assets/0773626f-5ea7-41c1-91bc-a6a32f664cf9" />


The dashboard includes multiple user-configurable inputs for exploring the graph:

📊 Influencer Ranking Table
- Sorted by hybrid importance score
- Searchable & filterable

🧭 Graph Embedding Map (t-SNE plot)
- Colored by influence score
- Reveals clusters and role communities

🕸 Network Graph Visualization
- Highlights top influencers
- Shows connectivity patterns

📈 User-Level Analytics
- Embedding norm
- Personal centrality metrics

## 🚀 Run Webapp
```bash
https://influencer-dashboard-ngjge6r92tojrstnadudjm.streamlit.app/
```

## Tech Stack

- Python
- Streamlit
- Plotly
- NumPy \& Pandas
- Matplotlib \& Seaborn
- t-SNE (Sklearn)
- NetworkX
- PyTorch Geometric
  

