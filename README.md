# ASFL: Anti-fragile Split Federated Learning

Official implementation of ASFL on IoMT data (WESAD, BIDMC, MHEALTH).

### Installation

```bash
pip install -r requirements.txt
```



## Dataset Setup

1. Download the WESAD, BIDMC, and MHEALTH datasets from their official sources.



1. Place them in the `data/` directory with the following structure:

```text
data/WESAD/WESAD/
data/BIDMC/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc\_csv/
data/MHEALTHDATASET/
```

3. Run the experiments (see [Reproducing Results](#reproducing-results) below).



## Repository Structure

```
ASFL/
├── \_\_init\_\_.py              # Package exports
├── config.py                # Configuration
├── data.py                  # Data loaders
├── models.py                # Neural networks
├── attacks.py               # Inversion \& PGD attacks
├── evaluation.py            # Metrics
├── trainer\_plain.py         # Plain SFL trainer
├── trainer\_ressfl.py        # ResSFL trainer
├── trainer\_asfl.py          # ASFL trainer
├── experiments.py           # Experiment orchestration
├── main.py                  # CLI entry point
├── utils.py                 # Utilities
├── data/
│   ├── WESAD/
│   │   └── WESAD/           ← Place WESAD .pkl files here
│   ├── BIDMC/
│   │   └── bidmc-ppg-and-respiration-dataset-1.0.0/
│   │       └── bidmc\_csv/   ← Place BIDMC CSV files here
│   └── MHEALTHDATASET/      ← Place MHEALTH .log files here
├── results/                 ← Experiment outputs
├── logs/                    ← Training logs
├── requirements.txt
└── README.md
```



## Reproducing Results

### Table 2: Hyperparameter Search

**WESAD:**

```bash
bash table2\_wesad\_iid.sh
bash table2\_wesad\_noniid.sh
```

**BIDMC:**

```bash
bash table2\_bidmc\_iid.sh
bash table2\_bidmc\_noniid.sh
```

**MHEALTH:**

```bash
bash table2\_mhealth\_iid.sh
bash table2\_mhealth\_noniid.sh
```

### Table 3: Hyperparameter Sensitivity and Training Stability

Mean ± standard deviation across the 10 sampled configurations per method from Table 2 (IID and Non-IID α = 0.5).

### Table 4: Reconstruction Resistance

Reconstruction resistance across the inversion-attacker hierarchy, using the best configurations from Table 2.

### Table 5: Ablation Studies

**MHEALTH:**

```bash
bash run\_table5\_mhealth.sh
```

**BIDMC:**

```bash
bash run\_table5\_bidmc.sh
```

**WESAD:**

```bash
bash run\_table5\_wesad\_iid.sh
bash run\_table5\_wesad\_noniid.sh
```

### Citation

If you use this code in your research, please cite the corresponding paper.

## Acknowledgements

The experiments in this repository were conducted using the High Performance Computing (HPC) cluster.



## License

