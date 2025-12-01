# Bayesian Integration of PPI Scores (PrePPI-SM + ZEPPI + D-Script-TT)

This repository contains a naïve Bayesian model for integrating multiple protein–protein interaction (PPI) scores—**PrePPI-SM**, **ZEPPI**, and **D-Script-TT**—into a single combined score. The model was trained on **human PPIs** and used to generate predictions for *E. coli* protein–protein interactions.

## Overview

Modern PPI predictors capture different biophysical and evolutionary signals. This project combines three complementary methods:

- **PrePPI-SM**: Structure-based PPI scoring
- **ZEPPI**: Sequence-based PPI interface coevolution scoring
- **D-Script-TT**: Deep-learning based protein-language model scoring for PPIs

These features are fused using a **discretized naïve Bayesian classifier**, which computes bin-wise likelihood ratios for each feature and multiplies them into a global combined score.

The model uses:

- Automatic binning per feature (Doane rule)
- Laplace smoothing for zero-count bins
- Likelihood ratio estimation from positive vs. negative training PPIs
- A highly imbalanced training set (1:1000 positives:negatives), reflecting realistic genome-wide PPI sparsity

Once trained, the Bayesian model can generalize across species and is applied here to score *E. coli* PPIs.

## Repository Structure

```
.
├── notebooks/
│   ├── Train_on_human.ipynb
│   ├── Test_on_ecoli_DEMO.ipynb
│
├── src/
│   ├── Bayes.py
│   └── evaluation.py
│
├── data/
│   ├── human_ZP_Bayes_ratio1000.pkl
│   ├── human_DS_Bayes_STRING.pkl
│
└── README.md
└── requirements.txt

```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/haiqingzhao/BayesianModel-for-Ecoli-PPI.git
cd BayesianModel-for-Ecoli-PPI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Training and Testing Data

Training the Bayesian model requires **three feature tables** and **two label datasets** for human PPIs. Testing the Bayesian model requires **three feature tables** for *E. coli* PPIs. Related methods have been previously published, here due to the large sizes of these genome-wide predictions, the direct PPI files will be available upon requests.

#### 3.1 Training Data (Human)

##### 3.1.1 Feature Tables (Human)

Prepare human PPI predictions separately from PrePPI, ZEPPI, and D-Script (TT). Each file must contain:

```
UniprotID_pair, score
```

##### 3.1.2 Training Labels (Human TP/TN)

Prepare TP and TN datasets. File format:

```
UniprotID_pair, label
```

Where `label = 1` (TP) or `label = 0` (TN).

#### 3.2 Testing Data (*E. coli*)

##### 3.1.1 Feature Tables (*E. coli*)

Prepare *E. coli* PPI predictions from PrePPI, ZEPPI, and D-Script (TT). Each file must contain:
```
UniprotID_pair, score
```

### 4. Run training

Open the training notebook:

```
Train_on_human.ipynb
```

It will need ~35GB memory to run it. The trained models of ZEPPI-LR and TT-LR on human — used in the paper — are provided as references. 

### 5. Run prediction for *E. coli*

Open:

```
Test_on_ecoli_DEMO.ipynb
```

## Contact

For questions, please reach out to:

**Haiqing Zhao hz2592@cumc.columbia.edu**  

