# SIT720 Task 11.1HD: Heart Attack Prediction

## Project Overview

This repository contains my implementation for SIT720 Task 11.1HD. The project reproduces and critically evaluates the machine learning methods presented in the research paper:

**M. Bhagat, A. Sharma, and P. Agarwal, “An efficient stacking-based ensemble technique for early heart attack prediction,” Multimedia Tools and Applications, vol. 84, no. 30, pp. 36351–36375 (2025). DOI: https://doi.org/10.1007/s11042-024-20064-7**

The project is divided into two main experimental parts:

1. **Part 1: Research Reproduction:** Reproduction of the six machine learning classifiers and stacking ensemble presented in the selected paper.
2. **Part 2: Proposed ML Solution:** Development of a leakage-resistant experimental protocol and an optimised soft-voting ensemble to address limitations identified during reproduction.

The accompanying technical report contains the complete methodology, results, comparisons, critical analysis and supporting literature.

---

## Repository Structure

```text
SIT720-Task-11.1HD/
│
├── README.md
├── requirements.txt
├── Task 11.1HD.ipynb
│
├── data/
│   └── heart.csv
│
└── report/
    └── Ananya Gupta- Task_11.1HD_Report.pdf
```

### Files

* **`Task 11.1HD.ipynb`** – Complete Jupyter Notebook containing Part 1 reproduction, Part 2 proposed solution, experiments, evaluation and visualisations.
* **`requirements.txt`** – Python package versions used for the implementation.
* **`data/heart.csv`** – Heart disease dataset used for the reproduction and proposed solution.
* **`report/Ananya Gupta- Task_11.1HD_Report.pdf`** – Technical research report containing the methodology, results and critical analysis.

---

# Part 1: Reproduction

## Dataset

The dataset used for the reproduction contains:

* **1,025 observations**
* **13 predictor variables**
* **1 binary target variable**

The target distribution contains:

* 526 positive observations
* 499 negative observations

The 13 predictors are:

* `age`
* `sex`
* `cp`
* `trestbps`
* `chol`
* `fbs`
* `restecg`
* `thalach`
* `exang`
* `oldpeak`
* `slope`
* `ca`
* `thal`

The target variable is:

* `target`

No missing values were identified in the dataset used for the experiments.

---

## Reproduced Machine Learning Models

The following classifiers from the selected paper were reproduced:

* Logistic Regression
* Naive Bayes
* K-Nearest Neighbours
* Decision Tree
* Random Forest
* XGBoost

A stacking ensemble containing the six classifiers was also implemented.

The evaluation metrics used were:

* Accuracy
* Precision
* Recall
* F1 Score
* Area Under the ROC Curve (AUC)

These match the evaluation metrics reported in the selected research paper.

---

## Reproduction Assumptions

Some implementation details were not completely specified in the selected paper. Therefore, reasonable assumptions were required.

The reproduction uses:

* an 80:20 stratified train/test split
* `random_state=42` where applicable
* `StandardScaler` for continuous features
* existing numerical codes for categorical features
* standard library parameters where classifier parameters were not fully specified
* five-fold cross-validation for stacking
* Logistic Regression as the stacking meta-classifier

The 80:20 split was selected because a test set containing 205 observations is consistent with several of the published accuracy values when converted to integer numbers of correct predictions.

These assumptions are explained and justified in greater detail in the technical report.

---

## Part 1 Results

The reproduced model results were:

| Model               | Accuracy | Precision | Recall | F1 Score |    AUC |
| ------------------- | -------: | --------: | -----: | -------: | -----: |
| Logistic Regression |   0.8146 |    0.7638 | 0.9238 |   0.8362 | 0.9297 |
| Naive Bayes         |   0.8293 |    0.8070 | 0.8762 |   0.8402 | 0.9043 |
| KNN                 |   0.8537 |    0.8571 | 0.8571 |   0.8571 | 0.9540 |
| Decision Tree       |   0.9854 |    1.0000 | 0.9714 |   0.9855 | 0.9857 |
| Random Forest       |   1.0000 |    1.0000 | 1.0000 |   1.0000 | 1.0000 |
| XGBoost             |   1.0000 |    1.0000 | 1.0000 |   1.0000 | 1.0000 |
| Stacking            |   1.0000 |    1.0000 | 1.0000 |   1.0000 | 1.0000 |

During analysis of the reproduced results, I identified 723 exact duplicate rows in the 1,025-row dataset.

For my Part 1 split, 202 of the 205 test observations had an identical predictor row in the training data**, corresponding to **98.54% exact predictor overlap.

This finding motivated the revised experimental methodology used in Part 2.

---

# Part 2: Proposed Machine Learning Solution

## Motivation

The near-perfect performance obtained during reproduction raised concerns about whether the Part 1 test set provided a strong estimate of generalisation.

My proposed solution therefore focuses on improving the experimental protocol as well as changing the ensemble methodology.

The proposed approach:

1. removes exact duplicate observations before splitting the data;
2. creates a stratified training and held-out test split;
3. ensures zero exact predictor overlap between the training and test sets;
4. performs preprocessing inside cross-validation pipelines;
5. restricts model selection and hyperparameter optimisation to the training data;
6. keeps the held-out test set untouched until final evaluation; and
7. combines optimised Logistic Regression and Random Forest models using soft voting.

This represents a methodological change rather than simply replacing one classifier or changing a single hyperparameter.

---

## Dataset Preparation

After removing exact duplicate observations:

```text
Original observations: 1025
Duplicate observations removed: 723
Unique observations: 302
```

The deduplicated target distribution is:

```text
Positive: 164
Negative: 138
```

A stratified 80:20 split produces:

```text
Training observations: 241
Held-out test observations: 61
```

The exact predictor overlap between the Part 2 training and held-out test sets is:

```text
0 / 61 = 0%
```

---

## Validation Strategy

Model development is performed only using the 241 training observations.

A five-fold `StratifiedKFold` procedure is used with:

```python
shuffle=True
random_state=42
```

Preprocessing is included inside the machine learning pipelines so that the scaler is fitted separately within each training fold.

The 61-observation held-out test set is not used for:

* baseline model comparison
* model selection
* hyperparameter optimisation
* cross-validation

It is used only after the final model has been selected.

---

## Baseline Cross-Validation Results

| Model               | Mean Accuracy | Mean Precision | Mean Recall | Mean F1 | Mean AUC |
| ------------------- | ------------: | -------------: | ----------: | ------: | -------: |
| Logistic Regression |        0.8420 |         0.8199 |      0.9234 |  0.8661 |   0.9149 |
| Random Forest       |        0.8257 |         0.8238 |      0.8778 |  0.8466 |   0.9097 |
| Naive Bayes         |        0.8172 |         0.8244 |      0.8624 |  0.8378 |   0.8950 |
| XGBoost             |        0.7964 |         0.8195 |      0.8160 |  0.8117 |   0.8884 |
| KNN                 |        0.8213 |         0.8026 |      0.9080 |  0.8481 |   0.8622 |
| Decision Tree       |        0.8007 |         0.8064 |      0.8393 |  0.8193 |   0.7969 |

Logistic Regression and Random Forest were selected for optimisation based primarily on their cross-validated AUC performance.

---

## Hyperparameter Optimisation

### Logistic Regression

The following values were evaluated:

```text
C = [0.01, 0.1, 1, 10, 100]
class_weight = [None, "balanced"]
```

Best configuration:

```text
C = 1
class_weight = balanced
```

Mean cross-validation AUC:

```text
0.9156
```

### Random Forest

The following parameters were tuned:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`

Best configuration:

```text
n_estimators = 100
max_depth = None
max_features = sqrt
min_samples_leaf = 4
min_samples_split = 2
```

Mean cross-validation AUC:

```text
0.9190
```

---

## Proposed Soft-Voting Ensemble

The final proposed model combines the optimised Logistic Regression and Random Forest models using probability-based soft voting.

Five-fold cross-validation produced:

| Metric                 | Result |
| ---------------------- | -----: |
| Accuracy               | 0.8461 |
| Precision              | 0.8387 |
| Recall                 | 0.9080 |
| F1 Score               | 0.8680 |
| AUC                    | 0.9254 |
| AUC Standard Deviation | 0.0337 |

---

## Final Held-Out Test Results

After all model selection and optimisation were complete, the proposed ensemble was fitted using the complete 241-observation training set and evaluated once on the 61-observation held-out test set.

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 0.7869 |
| Precision | 0.8125 |
| Recall    | 0.7879 |
| F1 Score  | 0.8000 |
| AUC       | 0.8777 |

Confusion matrix:

```text
[[22, 6],
 [ 7,26]]
```

The held-out test set contains zero exact predictor overlap with the training set.

---

## Controlled Ensemble Comparison

To determine whether the soft-voting architecture contributed beyond the change in evaluation protocol, I also compared it with a paper-style stacking ensemble using the same:

* deduplicated training data
* five cross-validation folds
* preprocessing protocol
* evaluation procedure

| Model                | Mean Accuracy | Mean Precision | Mean Recall | Mean F1 | Mean AUC | AUC SD |
| -------------------- | ------------: | -------------: | ----------: | ------: | -------: | -----: |
| Paper-style Stacking |        0.8297 |         0.8263 |      0.8855 |  0.8516 |   0.9103 | 0.0387 |
| Proposed Soft Voting |        0.8461 |         0.8387 |      0.9080 |  0.8680 |   0.9254 | 0.0337 |

Under the same experimental conditions, the proposed soft-voting ensemble increased mean AUC by 0.0151 and mean Accuracy by 0.016.

This is interpreted as a modest empirical improvement rather than evidence that soft voting is universally superior to stacking.

---

# Installation

## 1. Clone the Repository

Clone this repository using Git:

```bash
git clone https://github.com/ananyag270903/SIT720-Task-11.1HD.git
```

Then move into the project directory:

```bash
cd SIT720-Task-11.1HD
```

Alternatively, the repository can be downloaded as a ZIP file from GitHub.

---

## 2. Create a Virtual Environment

A separate virtual environment is recommended to avoid conflicts with existing Python packages.

Using Python:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

---

## 3. Install Dependencies

Install the required packages using:

```bash
python3 -m pip install -r requirements.txt
```

The implementation was developed and tested using Python 3.12.

The main package versions are:

```text
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
scikit-learn==1.4.2
xgboost==3.4.1
kagglehub==1.0.2
jupyter==1.0.0
```

The complete list is also provided in `requirements.txt`.

---

# Running the Notebook

Start Jupyter Notebook from the repository directory:

```bash
jupyter notebook
```

Open:

```text
Task 11.1HD.ipynb
```

Before reproducing the results, restart the kernel and select:

```text
Run All
```

The notebook is designed to execute sequentially from beginning to end.

The submitted version was tested using a clean kernel to confirm that all cells execute successfully and that the reported outputs are visible.

---

# Reproducing the Results

To reproduce the submitted experiments:

1. Install the packages listed in `requirements.txt`.
2. Ensure the dataset is available in the repository's `data/` directory.
3. Open `Task 11.1HD.ipynb`.
4. Restart the Jupyter kernel.
5. Run all cells from top to bottom.
6. Confirm that the Part 1 model results are generated.
7. Confirm that the duplicate and train-test overlap analysis is generated.
8. Continue through the Part 2 deduplication and leakage-resistant split.
9. Allow the hyperparameter searches and cross-validation experiments to complete.
10. Confirm the final soft-voting evaluation, confusion matrix, ROC curve and controlled stacking comparison.

Some model training cells, particularly the Random Forest grid search and ensemble cross-validation, may take longer than other cells to execute.

---

# Expected Key Results

A successful complete execution should reproduce the main findings below.

### Part 1

```text
Dataset size: 1025 observations
Exact duplicate rows: 723
Part 1 training size: 820
Part 1 test size: 205
Exact predictor overlap: 202 / 205 (98.54%)
Reproduced stacking accuracy: 1.0000
Reproduced stacking AUC: 1.0000
```

### Part 2

```text
Unique observations after deduplication: 302
Training observations: 241
Held-out test observations: 61
Exact predictor overlap: 0%
Soft-voting mean CV AUC: 0.9254
Soft-voting CV AUC SD: 0.0337
Held-out Accuracy: 0.7869
Held-out AUC: 0.8777
```

### Controlled Comparison

```text
Paper-style stacking mean AUC: 0.9103
Proposed soft-voting mean AUC: 0.9254
AUC difference: +0.0151
```

---

# Reproducibility Notes

Random seeds are fixed where applicable to improve reproducibility.

The main random state used throughout the implementation is:

```python
random_state=42
```

The Part 2 cross-validation procedure also uses:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Minor differences may occur if the notebook is executed using substantially different Python or package versions. For this reason, the exact package versions used for the submitted implementation are provided in `requirements.txt`.

---

# Technical Report

The complete research report is available at:

```text
report/SIT720_Task_11_1HD_Report.pdf
```

The report contains:

* research problem and motivation
* state-of-the-art approaches
* research gap
* dataset and feature set
* Part 1 reproduction methodology
* reproduction assumptions
* published versus reproduced results
* critical analysis of discrepancies
* Part 2 proposed methodology
* hyperparameter optimisation
* cross-validation experiments
* held-out evaluation
* controlled stacking versus soft-voting comparison
* limitations and practical implications
* supporting literature
* Video presentation and Github repository links 
* IEEE-formatted references

---

# Video Presentation

The accompanying video presentation demonstrates the reproduction, proposed solution, selected implementation outputs, comparative results and reflection.

**Video link:** https://deakin.au.panopto.com/Panopto/Pages/Viewer.aspx?id=4d2bf71f-bbe8-48e8-9246-b4d100a50bec 
---

## Reference

M. Bhagat, A. Sharma, and P. Agarwal, “An efficient stacking-based ensemble technique for early heart attack prediction,” Multimedia Tools and Applications, vol. 84, no. 30, pp. 36351–36375 (2025). DOI: https://doi.org/10.1007/s11042-024-20064-7  

