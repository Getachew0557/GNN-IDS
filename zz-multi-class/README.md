# Multiclass Attack Graph Classification

A comprehensive MLOps pipeline for multiclass classification on attack graphs using graph neural networks and traditional ML approaches.

## Project Structure

```bash
multiclass/
├── data/
│   │   └── public/
│   │       └── CICD-IDS2017.csv
│  
└── attack_graphs/
│       └── AttackGraph.dot
├── src/
│   ├── __init__.py
│   ├── ag_utils.py
│   ├── data_utils.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── experiment.ipynb
├── mlops/
│   ├── config.yaml
│   ├── mlflow_setup.py
│   └── deployment/
│       └── serve_model.py
├── tests/
│   ├── __init__.py
│   ├── test_ag_utils.py
│   └── test_data_utils.py
├── requirements.txt
└── README.md
```


## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. Run experiments:

   ```bash
    python src/train.py
   ```
3. Launch MLFlow UI:

    ```bash
    mlflow ui --backend-store-uri mlruns/
    ```
4. Run tests:

    ```bash
    pytest tests/
    ```
## Features
- Multiple GNN architectures (GCN, GAT, GraphSAGE, TAD-GAT)
- Comprehensive MLOps with MLFlow tracking
- Synthetic and public dataset support
- Automated experiment tracking
- Model deployment ready


