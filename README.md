# Predicting Nucleon Separation Energies with Machine Learning

> **Note:** This project is an extension and improvement upon the work presented in the Master's thesis "From Nucleus to Algorithms: Predicting Neutron separation energy with Deep Neural Network (DNN)" by Sourav Gayen (IIT ISM Dhanbad, 2025). The DNN model serves as a baseline for comparison against the improved XGBoost model developed here.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning project to predict nucleon separation energies ($S_n$, $S_{2n}$, and $S_p$) by implementing and comparing a Deep Neural Network (DNN) and an XGBoost model.


## 🎯 Motivation & Goal

The prediction of nucleon separation energies is a fundamental challenge in nuclear physics, crucial for understanding nuclear structure, stability, and the limits of the nuclear chart. Accurate predictions are vital not only for nuclear theory but also for astrophysical applications, such as modeling the r-process nucleosynthesis.

While traditional theoretical models have been successful, they often struggle to extrapolate into exotic, data-scarce regions near the drip lines. This project tackles this challenge by developing and comparing two distinct, powerful machine learning approaches:

1.  **Deep Neural Network (DNN)**: A baseline model designed to learn complex, non-linear relationships directly from the data.
2.  **XGBoost**: A state-of-the-art gradient boosting model, used here to improve upon the baseline performance.

The goal is to not only achieve high predictive accuracy but also to compare the performance and interpretability of these two different modeling paradigms on a complex physics problem.

## 💾 Dataset

The model is trained on experimental data sourced from the **NuDat database**, maintained by the National Nuclear Data Center (NNDC).

The input features were carefully selected and engineered to provide the models with physically meaningful information:

* **Fundamental Features**: Proton number (Z), Neutron number (N), Mass number (A).
* **Engineered Features**: To capture complex nuclear effects, the following were included:
    * Isospin Asymmetry: $(N-Z)/A$
    * Neutron-to-Proton Ratio: $n/p$
    * Mass Scaling Term: $A^{2/3}$
    * Pairing Effects: Binary flags for odd/even Z and N numbers.
    * Shell Structure: Valence nucleon counts ($V_z$, $V_n$) and categorical indicators for shell regions ($Z_{shell}$, $N_{shell}$).

All features were standardized using **Z-score normalization** before being used in the DNN.

## 🛠️ Methodology

This project implements two distinct models for a comprehensive comparative analysis.

### 1. Deep Neural Network (DNN) - The Baseline

The baseline model is a feedforward neural network built with **PyTorch**. Its architecture is directly inspired by the work presented in the Master's thesis "From Nucleus to Algorithms: Predicting Neutron separation energy with Deep Neural Network (DNN)" by Sourav Gayen (IIT ISM Dhanbad, 2025). It consists of multiple dense hidden layers with `ReLU` and `Tanh` activations and utilizes L2 regularization to prevent overfitting. This implementation serves as a benchmark to evaluate the XGBoost model.

### 2. XGBoost - The Improvement

An implementation of **eXtreme Gradient Boosting**, a powerful and efficient tree-based ensemble method. XGBoost builds a strong predictive model by sequentially adding weak learner models (decision trees), with each new tree correcting the errors of the previous ones. It is highly effective for structured/tabular data and often provides superior performance.

### Interpretability

For both models, **SHapley Additive exPlanations (SHAP)** analysis is used to understand feature importance and connect the models' predictions back to underlying nuclear physics principles.

## 📊 Results & Model Comparison

Both models achieved high predictive accuracy, demonstrating the viability of ML for this task. A comparison of their performance measure in terms of Mean Square Error (MSE) on test set is below:

| Model |   $S_n$   |   $S_p$   |   $S_{2n}$   |
| :---- | :-------- | :-------- | :-------- |
| DNN   | 0.337 MeV | 0.385 MeV | 0.760 MeV |
| XGBoost| 0.178 MeV | 0.217 MeV | 0.223 MeV |

#### SHAP Feature Importance Summary (XGBoost)

SHAP analysis for the XGBoost model confirms that physically significant features like **isospin asymmetry** and **shell indicators** are the primary drivers of its predictions, reinforcing the model's physical consistency.
![SHAP Summary Plot](results/figures/shap_summary_xgb.png)
*(Note: Replace with the actual path to your saved plot in the repository.)*

## 🚀 How to Use

### Prerequisites

-   Python 3.8+
-   PyTorch
-   XGBoost
-   NumPy, Pandas, Scikit-learn
-   Matplotlib, Seaborn
-   SHAP

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
    cd <your-repo-name>
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

Use the main training script. You can specify which model to run via command-line arguments.

1.  **Train the DNN Model**:
    ```bash
    python src/train.py --model dnn --target Sn --epochs 200
    ```

2.  **Train the XGBoost Model**:
    ```bash
    python src/train.py --model xgboost --target Sn
    ```
