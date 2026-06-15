# Time Series Analysis and Forecasting - Group 04

This repository contains the codebase and reports for the Time Series Analysis and Forecasting project by Group 04.

## Repository Structure

```text
time-series-group-04/
├── README.md                           # Project documentation
├── papers/                             # Literature and research papers
│   ├── paper_01.md
│   ├── paper_02.md
│   └── paper_03.md
├── data/                               # Dataset directory (git-ignored except keep-files)
│   ├── raw/
│   └── processed/
├── notebooks/                          # Jupyter Notebooks for step-by-step analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_models.ipynb
│   └── 04_evaluation.ipynb
├── src/                                # Source code for modularized functions
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   └── evaluation.py
├── figures/                            # Generated plots and figures
├── results/                            # Saved models, metrics, and outputs
│   └── metrics.csv
├── report/                             # Final project report
│   └── final_report.md
├── requirements.txt                    # Project dependencies
└── .gitignore                          # Files to exclude from Git
```

## Getting Started

### Prerequisites

To install dependencies, run:

```bash
pip install -r requirements.txt
```

### Running the Project

1. **Data Exploration**: Run the Jupyter Notebook `notebooks/01_data_exploration.ipynb` or execute custom exploration scripts.
2. **Feature Engineering**: Use `notebooks/02_feature_engineering.ipynb` to construct features, which are implemented modularly in `src/features.py`.
3. **Model Training**: Train models via `notebooks/03_models.ipynb` using implementations in `src/models.py`.
4. **Evaluation**: Evaluate outputs in `notebooks/04_evaluation.ipynb` with routines in `src/evaluation.py`.
