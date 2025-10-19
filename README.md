# Predicting Nuclear Separation Energies with Deep Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning project to predict nucleon separation energies ($S_n$, $S_{2n}$, and $S_p$) using a physics-informed feature set and a deep neural network architecture implemented in PyTorch.

## 🎯 Motivation & Goal

The prediction of nucleon separation energies is a fundamental challenge in nuclear physics. [cite_start]These values are crucial for understanding nuclear structure, stability, and the limits of the nuclear chart, particularly in the exotic, data-scarce regions near the neutron and proton drip lines[cite: 36]. [cite_start]Accurate predictions are vital not only for nuclear theory but also for astrophysical applications, such as modeling the r-process nucleosynthesis responsible for creating heavy elements[cite: 217, 519].

While traditional theoretical models have been successful, they often struggle to extrapolate into these unexplored regions. This project aims to bridge that gap by developing a **data-driven, physics-informed deep learning model** capable of:

1.  Accurately predicting neutron ($S_n$), two-neutron ($S_{2n}$), and proton ($S_p$) separation energies.
2.  Generalizing its predictions to nuclei far from the valley of stability.
3.  Providing physically interpretable results that align with established nuclear theory.

## 💾 Dataset

[cite_start]The model is trained on experimental data sourced from the **NuDat database**, maintained by the National Nuclear Data Center (NNDC)[cite: 29, 271].

The input features were carefully selected and engineered to provide the model with physically meaningful information:

* [cite_start]**Fundamental Features**: Proton number (Z), Neutron number (N), Mass number (A). [cite: 300]
* **Engineered Features**: To capture complex nuclear effects, the following were included:
    * [cite_start]Isospin Asymmetry: $(N-Z)/A$ [cite: 304]
    * [cite_start]Neutron-to-Proton Ratio: $n/p$ [cite: 94]
    * [cite_start]Mass Scaling Term: $A^{2/3}$ [cite: 305]
    * [cite_start]Pairing Effects: Binary flags for odd/even Z and N numbers. [cite: 306]
    * [cite_start]Shell Structure: Valence nucleon counts ($V_z$, $V_n$) and categorical indicators for shell regions ($Z_{shell}$, $N_{shell}$). [cite: 307, 308]

[cite_start]All features were standardized using **Z-score normalization** before being fed into the network to ensure stable training[cite: 321].

## 🛠️ Methodology

The core of this project is a **Deep Neural Network (DNN)** built with Python and PyTorch.

* [cite_start]**Architecture**: The model consists of several dense (fully-connected) hidden layers using a mix of `ReLU` and `Tanh` activation functions. [cite: 623]
* [cite_start]**Regularization**: To prevent overfitting and improve generalization, **L2 regularization** is applied to the weights of the hidden layers. [cite: 39, 451]
* [cite_start]**Optimization**: The network is trained using the **Adam optimizer** to minimize the **Mean Squared Error (MSE)** between the predicted and true separation energies. [cite: 39, 441]
* [cite_start]**Interpretability**: After training, **SHapley Additive exPlanations (SHAP)** analysis is used to understand the model's decisions and verify that its feature importances align with physical principles. [cite: 43, 497]

## 📊 Results

The model achieves high predictive accuracy across all three targets. For the neutron separation energy ($S_n$) prediction, the key results on the test set are:
* [cite_start]**Mean Absolute Error (MAE)**: 0.210 MeV [cite: 734]
* [cite_start]**Prediction Accuracy (>95%)**: Achieved for nuclei within a ±0.5 MeV tolerance. [cite: 490, 736]

Key findings include:
* [cite_start]**Physical Consistency**: The model successfully learned and reproduced known physical phenomena without being explicitly programmed to do so, including the effects of **shell closures**, **pairing correlations**, and smooth trends across isotopic chains. [cite: 42]
* [cite_start]**Feature Importance**: SHAP analysis confirms that the most influential features are **isospin asymmetry** ($(N-Z)/A$), **neutron-to-proton ratio**, and **shell-related indicators**—perfectly aligning with established nuclear theory. [cite: 44, 702]

#### True vs. Predicted Values ($S_n$)
![True vs Predicted Plot](results/figures/true_vs_predicted_Sn.png)
*(This is a placeholder. You should replace this with a link to your actual plot in the repository.)*

#### SHAP Feature Importance Summary
![SHAP Summary Plot](results/figures/shap_summary_Sn.png)
*(This is a placeholder. You should replace this with a link to your actual plot in the repository.)*

## 🚀 How to Use

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SHAP

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

1.  **Train a Model**:
    Use the main training script to train a new model. You can specify the target observable.
    ```bash
    python src/train.py --target Sn --epochs 200 --batch_size 32
    ```

2.  **Make Predictions**:
    Use a prediction script to get the separation energy for a given nucleus.
    ```bash
    python src/predict.py --target Sn --Z 8 --N 8
    ```
