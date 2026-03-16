# Italian Municipal Facebook Topic Modeling

This repository contains the **open-source pipeline of a private research project** analyzing communication patterns of Italian municipalities and local politicians on Facebook (2008–2024).

The original dataset contains **millions of posts stored in a 2.4GB Parquet file**.  
To respect data privacy and platform policies, **raw data is not included in this repository**. Instead, the repository provides:

- the **complete reproducible pipeline**
- **configurations and infrastructure**
- **aggregated results and visualizations**

This allows the methodology and results to be shared **without leaking sensitive raw data**.

---

# Project Overview

The project builds a scalable **topic modeling pipeline for large-scale social media text** using modern NLP and data engineering practices.

The pipeline processes Facebook posts from municipalities and politicians, extracts semantic representations, and discovers latent themes in political communication over time.

The repository is structured to demonstrate **production-ready data science workflows**, including configuration management, reproducibility, and modular code organization.

---

# Techniques Used

The pipeline combines data engineering, NLP, and machine learning techniques:

- **SQL-first Parquet querying with DuckDB**
- **Batch / streaming processing** to control memory usage
- **Social media text cleaning and normalization**
- **Sentence-transformer embeddings**
- **Dimensionality reduction (UMAP)**
- **Density-based clustering (HDBSCAN)**
- **BERTopic topic modeling**
- **Config-driven pipeline runs**
- **Dockerized reproducibility**

These techniques allow efficient topic modeling on **hundreds of thousands to millions of posts** on a consumer laptop.

---

# High-Level Pipeline

The workflow is designed as a modular NLP pipeline.

```
Parquet dataset
      ↓
DuckDB SQL profiling
      ↓
Data sampling & filtering
      ↓
Social media text cleaning
      ↓
SentenceTransformer embeddings
      ↓
UMAP dimensionality reduction
      ↓
HDBSCAN clustering
      ↓
BERTopic topic extraction
      ↓
Topic evaluation & visualization
```

---

# Repository Structure

```
topic-model-italian-facebook
│
├── configs/                  # Configuration files for experiments
│
├── data/
│   ├── raw/                  # Raw dataset location (not included)
│   ├── interim/              # Intermediate datasets
│   └── processed/            # Cleaned text and modeling inputs
│
├── models/                   # Saved BERTopic models
│
├── notebooks/                # Exploratory analysis and visualization
│
├── reports/                  # Figures and result summaries
│
├── scripts/                  # Pipeline entry-point scripts
│
├── src/
│   ├── ingest/               # Data profiling and sampling
│   ├── preprocessing/        # Text cleaning
│   ├── modeling/             # Embeddings and topic modeling
│   ├── evaluation/           # Topic evaluation
│   └── utils/                # Config and helper utilities
│
├── Dockerfile                # Reproducible environment
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# Key Features

### Scalable processing
The pipeline handles **large Parquet datasets** using DuckDB SQL queries instead of loading everything into memory.

### Memory-efficient NLP
Embedding generation and preprocessing are performed **in batches**, allowing the pipeline to run on a laptop with limited RAM.

### Config-driven experiments
All runs are controlled via YAML configuration files, enabling reproducible experiments without modifying code.

### Reproducibility
The project supports:

- Python virtual environments
- pinned dependencies
- Docker containers

This ensures results can be reproduced across machines.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/topic-model-italian-facebook.git
cd topic-model-italian-facebook
```

Create environment:

```bash
python -m venv .venv
```

Activate environment:

Windows:

```bash
.venv\Scripts\activate
```

Linux / Mac:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

The pipeline is organized into modular scripts.

Example workflow:

### 1. Profile dataset

```bash
python -m scripts.01_profile_sql
```

### 2. Sample dataset

```bash
python -m scripts.02_sample_dataset
```

### 3. Clean text

```bash
python -m scripts.03_clean_text
```

### 4. Fit topic model

```bash
python -m scripts.04_fit_topic_model
```

### 5. Assign topics

```bash
python -m scripts.05_assign_topics
```

### 6. Evaluate topics

```bash
python -m scripts.06_evaluate_topics
```

---

# Example Output

The pipeline produces:

- discovered topic clusters
- topic keywords
- document-topic assignments
- topic evolution over time

Example topic:

```
Topic: Urban Infrastructure
Keywords: mobilità, traffico, trasporti, lavori, strada
```

---

# Why This Project

This repository demonstrates skills relevant to **data scientist and machine learning engineering roles**, including:

- large-scale text processing
- production-style NLP pipelines
- reproducible research workflows
- modular project architecture
- scalable topic modeling

---

# Data Availability

The original dataset contains Facebook posts collected from public pages of municipalities and politicians.

To comply with data usage policies and privacy considerations:

- raw posts are **not distributed**
- only **aggregated results and derived artifacts** are included

Users can reproduce the pipeline using their own datasets with similar schema.

---

# Future Improvements

Potential extensions include:

- dynamic topic modeling over time
- temporal trend analysis
- topic sentiment analysis
- multilingual political communication analysis

---

# License

MIT License

---

# Contact

If you have questions about the methodology or pipeline design, feel free to open an issue.