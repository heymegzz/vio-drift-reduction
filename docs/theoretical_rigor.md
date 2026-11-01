# Theoretical Rigor: Mathematical Formulations & Justifications

## 1. Problem Formulation

The objective is to estimate the instantaneous velocity residual vector $\mathbf{y}_t \in \mathbb{R}^3$ between the UAV's PX4 Extended Kalman Filter (EKF2) velocity estimate $\mathbf{v}_t^{\text{PX4}} \in \mathbb{R}^3$ and the true GPS ground velocity $\mathbf{v}_t^{\text{GPS}} \in \mathbb{R}^3$ in North-East-Down (NED) frame:

$$\mathbf{y}_t = \mathbf{v}_t^{\text{PX4}} - \mathbf{v}_t^{\text{GPS}} = \begin{bmatrix} \Delta v_{N,t} \\ \Delta v_{E,t} \\ \Delta v_{D,t} \end{bmatrix}$$

Using exclusively IMU-derived features $\mathbf{x}_t \in \mathbb{R}^D$ extracted from triaxial specific force $\mathbf{a}_t \in \mathbb{R}^3$ and angular rates $\boldsymbol{\omega}_t \in \mathbb{R}^3$ measured in the body frame:

$$\hat{\mathbf{y}}_t = f(\mathbf{x}_t; \boldsymbol{\theta})$$

At test/inference time during a GNSS outage, the corrected velocity $\tilde{\mathbf{v}}_t$ is computed via residual subtraction:

$$\tilde{\mathbf{v}}_t = \mathbf{v}_t^{\text{PX4}} - \hat{\mathbf{y}}_t$$

---

## 2. Model 1: Multi-Output Ridge Regression (L2-Regularized Linear Baseline)

### Mathematical Formulation
Ridge regression minimizes the sum of squared prediction errors subject to an $L_2$ penalty on regression weights:

$$\min_{\mathbf{W}, \mathbf{b}} \frac{1}{2N} \sum_{i=1}^N \|\mathbf{y}_i - (\mathbf{W}^T \mathbf{x}_i + \mathbf{b})\|_2^2 + \frac{\alpha}{2} \|\mathbf{W}\|_F^2$$

where:
- $\mathbf{W} \in \mathbb{R}^{D \times 3}$ is the weight matrix for the 3 target axes.
- $\mathbf{b} \in \mathbb{R}^3$ is the bias vector.
- $\|\mathbf{W}\|_F = \sqrt{\sum_{j=1}^D \sum_{k=1}^3 W_{jk}^2}$ denotes the Frobenius norm.
- $\alpha > 0$ controls the regularization strength.

The closed-form analytical solution for centered data is:

$$\mathbf{W}^* = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{Y}$$

### Theoretical Justification
1. **Multicollinearity Suppression:** Because engineered IMU features (rolling averages across 10, 50, and 100 sample windows) exhibit significant inter-feature correlation, $\mathbf{X}^T\mathbf{X}$ is ill-conditioned. The ridge term $\alpha \mathbf{I}$ guarantees strict positive-definiteness and bounds the parameter variance.
2. **Deterministic Linear Baseline:** Provides an essential reference for evaluating whether non-linear combinations of IMU signals are strictly required to resolve drift.
3. **Ultra-Low Latency:** Inference requires a single matrix-vector multiplication ($\mathcal{O}(D)$ per step), making it deployable on low-cost microcontrollers.

---

## 3. Model 2: Random Forest Regressor (Bagged Decision Tree Ensemble)

### Mathematical Formulation
Random Forest constructs an ensemble of $B$ independent de-correlated regression trees $\{T_b\}_{b=1}^B$, trained on bootstrap samples $\mathcal{D}_b^*$ of the training data $\mathcal{D}$:

$$\hat{\mathbf{y}}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$

At each node split in individual trees, only a random subset $m = \lfloor\sqrt{D}\rfloor$ of the available features is considered. Split candidate selection maximizes variance reduction (mean squared error reduction):

$$\Delta \mathcal{L}(s, j) = \text{MSE}(\mathcal{S}) - \left[ \frac{|\mathcal{S}_L|}{|\mathcal{S}|}\text{MSE}(\mathcal{S}_L) + \frac{|\mathcal{S}_R|}{|\mathcal{S}|}\text{MSE}(\mathcal{S}_R) \right]$$

### Theoretical Justification
1. **Variance Reduction via Bagging:** For $B$ trees with pairwise correlation $\rho$ and individual variance $\sigma^2$, the ensemble variance satisfies:
   $$\text{Var}(\hat{y}_{\text{RF}}) = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2$$
   By randomly subsampling features at each split, $\rho$ is minimized, yielding far lower variance than a single complex decision tree without increasing bias.
2. **Non-Linear Dynamics Modeling:** Captures high-order interactions between angular velocity spikes and linear acceleration components (e.g., centripetal accelerations during turns) without needing explicit parametric formulas.
3. **Inherent Feature Importance:** Calculates mean decrease in node impurity (Gini importance) to identify which sensor axes and temporal windows dominate drift prediction.

---

## 4. Model 3: Histogram-Based Gradient Boosting (Boosted Tree Ensemble)

### Mathematical Formulation
Gradient Boosting constructs an additive sequence of $M$ decision trees, where each subsequent tree fits the negative gradient (pseudo-residuals) of the loss function $\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})$:

$$f_m(\mathbf{x}) = f_{m-1}(\mathbf{x}) + \eta h_m(\mathbf{x})$$

For squared error loss $\mathcal{L}(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$, the pseudo-residual at step $m$ for sample $i$ is:

$$r_{im} = -\left[\frac{\partial \mathcal{L}(y_i, f(x_i))}{\partial f(x_i)}\right]_{f=f_{m-1}} = y_i - f_{m-1}(x_i)$$

The base learner $h_m$ minimizes:

$$h_m = \arg\min_h \sum_{i=1}^N (r_{im} - h(\mathbf{x}_i))^2 + \lambda \Omega(h)$$

where $\eta \in (0, 1]$ is the shrinkage factor (learning rate) and $\lambda \Omega(h)$ applies $L_2$ regularization on leaf values.

### Theoretical Justification
1. **Targeted Bias Minimization:** Unlike bagging which primarily reduces variance, boosting sequentially targets hard-to-predict regions of the state space (e.g., sudden velocity changes during sharp UAV maneuvers).
2. **Histogram Binning Efficiency:** Features are discretized into 256 integer bins, reducing split-finding complexity from $\mathcal{O}(N \cdot D)$ to $\mathcal{O}(K \cdot D)$ ($K=256$). This yields 10x-20x training acceleration on large temporal flight logs.
3. **Regularized Leaf Weighting:** Incorporates $L_2$ penalty on leaf outputs to prevent overfitting to noisy high-frequency accelerometer vibrations.

---

## 5. Evaluation Protocol & Safeguards Against Leakage

1. **Log-Partition Separation:** No split is performed row-wise. Flight logs `log_51` and `log_53` serve as training, `log_52` for validation/early stopping, and `log_54` is held out for unbiased testing.
2. **Autocorrelation Protection:** Because IMU drift and velocity residuals exhibit autocorrelation over several seconds, random cross-validation artificially leaks future state information; whole-flight separation preserves real-world generalization conditions.
3. **Exclusion of GPS-Derived Features:** All GPS velocities, satellite counts, EPH/EPV quality metrics, and estimator validity flags are strictly quarantined from model inputs.

