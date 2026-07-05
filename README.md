# Model Drift Detection using Evidently AI

## Project Overview

This project demonstrates how to detect data drift in a Machine Learning model using Evidently AI.

A Logistic Regression model is trained on the Breast Cancer Wisconsin Dataset. Artificial drift is then introduced into the original dataset by adding Gaussian noise, shifting important numerical features, and flipping 10% of the target labels. Finally, Evidently AI compares the original and drifted datasets and generates an interactive HTML report highlighting the detected drift.

---

## Dataset

Breast Cancer Wisconsin Dataset (Scikit-learn)

- Samples: 569
- Features: 30
- Target Classes:
  - 0 → Malignant
  - 1 → Benign

---

## Project Structure

```
Model_Drift_Evidently/

│
├── data/
│   ├── reference.csv
│   └── current.csv
│
├── models/
│   └── model.pkl
│
├── reports/
│   └── model_drift_report.html
│
├── src/
│   ├── train.py
│   ├── simulate_drift.py
│   └── drift_report.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Evidently AI
- Matplotlib
- Joblib

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Execution

### Step 1

```bash
python src/train.py
```

### Step 2

```bash
python src/simulate_drift.py
```

### Step 3

```bash
python src/drift_report.py
```

---

## Output

The project generates:

- Trained Model (`model.pkl`)
- Reference Dataset (`reference.csv`)
- Drifted Dataset (`current.csv`)
- Interactive HTML Drift Report (`model_drift_report.html`)

---

## Drift Simulation

The following drift techniques were used:

- Gaussian Noise
- Feature Shifting
- 10% Target Label Flipping

---

## Learning Outcomes

Through this project I learned:

- Machine Learning model training
- Saving trained models
- Dataset drift simulation
- Data drift detection using Evidently AI
- Professional project organization
- GitHub project management

---

## Author

Pari Dubey