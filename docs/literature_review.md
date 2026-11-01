# Literature Review: ML-Assisted IMU Drift Reduction for GNSS-Denied UAV Navigation

---

## 1. Introduction and Problem Context

Unmanned Aerial Vehicles (UAVs) rely on Inertial Measurement Units (IMUs)—comprising triaxial accelerometers and gyroscopes—as the primary sensors for attitude, velocity, and position estimation. In environments where Global Navigation Satellite System (GNSS) signals are unavailable or degraded—such as indoors, urban canyons, dense foliage, or under deliberate jamming—the navigation solution must rely heavily on inertial dead reckoning. However, low-cost Micro-Electro-Mechanical Systems (MEMS) IMUs suffer from inherent biases, scale-factor errors, axis misalignments, and stochastic noise processes including Angular Random Walk (ARW), Velocity Random Walk (VRW), and bias instability (Groves, 2013; Skog and Händel, 2006). Because velocity is obtained by single-integrating acceleration and position by double-integrating it, these small errors compound rapidly, leading to drift that can accumulate position errors of 1–10% of the distance traveled within minutes (El-Sheimy et al., 2020).

Traditional mitigation strategies employ model-based estimation filters—primarily the Extended Kalman Filter (EKF)—to fuse IMU predictions with external aiding sources such as GPS. The PX4 autopilot's EKF2 module, widely used in open-source UAV platforms, implements an error-state EKF that propagates 24 states including attitude, velocity, position, IMU biases, magnetic field components, and wind velocity (PX4 Documentation; Meier et al., 2015). When GPS is available, velocity and position innovations (residuals between predicted and measured states) drive the filter's correction step. During GPS outages, however, the filter operates in open-loop dead-reckoning mode, and the error covariance grows unboundedly. This fundamental limitation motivates the exploration of machine learning approaches that can learn the complex, non-linear mapping between raw IMU features and the estimator's velocity errors—effectively producing learned corrections that substitute for missing GPS updates.

---

## 2. Foundational Works on IMU Error Modeling and Inertial Navigation

### 2.1 Groves (2013) — Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems

- **Citation:** Groves, P. D. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems* (2nd ed.). Artech House.
- **Key Contribution:** Comprehensive textbook covering navigation mathematics, coordinate frames, inertial sensor error models, GNSS signal processing, Kalman filtering, and multi-sensor integration architectures (loosely coupled, tightly coupled, ultra-tight).
- **Relevance:** Provides the theoretical foundation for understanding how IMU errors propagate through strapdown navigation equations and why velocity/position errors grow during GNSS outages. The error-state Kalman filter formulation used in PX4 EKF2 is directly derived from the framework presented in this text.
- **Key Findings:** Systematic analysis showing that without external aiding, certain error states (heading drift, horizontal accelerometer biases) become unobservable, leading to unbounded position error growth.

### 2.2 Skog and Händel (2006) — Calibration of a MEMS Inertial Measurement Unit

- **Citation:** Skog, I. and Händel, P. (2006). "Calibration of a MEMS Inertial Measurement Unit." In *Proceedings of the XVII IMEKO World Congress*, Rio de Janeiro, Brazil.
- **Key Contribution:** Introduced a platform-free calibration methodology for MEMS IMUs that extracts deterministic error parameters (biases, scale factors, axis misalignment) using gravity-alignment constraints and nonlinear least-squares optimization.
- **Relevance:** Establishes the deterministic error model that underlies all subsequent work on IMU error compensation. The residual stochastic errors that remain after calibration are precisely what data-driven ML approaches seek to model.

### 2.3 Noureldin et al. (2013) — Fundamentals of Inertial Navigation, Sensor Technologies, and Integration

- **Citation:** Noureldin, A., Karamat, T. B., and Georgy, J. (2013). *Fundamentals of Inertial Navigation, Sensor Technologies, and Integration*. Springer.
- **Key Contribution:** Comprehensive treatment of MEMS sensor stochastic error modeling using Autoregressive (AR) and Gauss-Markov processes, Allan Variance analysis, and the integration of AI-based prediction modules (ANFIS, neural networks) with Kalman filters for GPS/INS bridging during outages.
- **Relevance:** Directly relevant to the project's approach of predicting navigation errors from IMU data. Noureldin's group pioneered the use of machine learning to map IMU features to INS error states during GPS outages.
- **Key Findings:** Demonstrated that ANFIS and neural networks trained on GPS-available periods could predict position/velocity errors during short-to-medium GPS outages (60–120 seconds), reducing drift by 50–80% compared to unaided INS.

---

## 3. Machine Learning for Inertial Navigation Error Reduction

### 3.1 Deep Inertial Odometry

#### 3.1.1 Chen et al. (2018) — IONet: Learning to Cure the Curse of Drift in Inertial Odometry

- **Citation:** Chen, C., Lu, X., Markham, A., and Trigoni, N. (2018). "IONet: Learning to Cure the Curse of Drift in Inertial Odometry." In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*.
- **Key Contribution:** Formulated inertial tracking as a sequence-to-sequence learning problem using deep recurrent neural networks (LSTMs/GRUs). Segmented inertial data into independent temporal windows to prevent error accumulation across windows.
- **Relevance:** Established the paradigm of using neural networks to learn motion patterns from IMU data without double integration—the conceptual precursor to predicting velocity residuals.
- **Key Findings:** Achieved significant trajectory accuracy improvements over traditional strapdown integration on indoor pedestrian datasets.

#### 3.1.2 Yan et al. (2018) — RIDI: Robust IMU Double Integration

- **Citation:** Yan, H., Shan, Q., and Furukawa, Y. (2018). "RIDI: Robust IMU Double Integration." In *Proceedings of the European Conference on Computer Vision (ECCV)*.
- **Key Contribution:** Used regression models to learn instantaneous velocity vectors from IMU history, then applied linear least-squares optimization to correct accelerometer biases before double integration.
- **Relevance:** Highly relevant—RIDI's approach of predicting velocity from IMU features is directly analogous to this project's prediction of velocity residuals. The key difference is that RIDI predicts absolute velocity while this project predicts velocity *errors* (PX4 − GPS).
- **Key Findings:** Demonstrated that simple data-driven velocity regression combined with physical correction could yield trajectory accuracies comparable to full Visual-Inertial Odometry (VIO).

#### 3.1.3 Herath et al. (2020) — RoNIN: Robust Neural Inertial Navigation in the Wild

- **Citation:** Herath, S., Yan, H., and Furukawa, Y. (2020). "RoNIN: Robust Neural Inertial Navigation in the Wild." In *Proceedings of IEEE International Conference on Robotics and Automation (ICRA)*.
- **Key Contribution:** Introduced one of the largest public inertial navigation datasets (40+ hours, 100 subjects) and systematically benchmarked multiple deep learning architectures (ResNet-based 1D CNNs, LSTMs, TCN) for supervised velocity regression from raw IMU data.
- **Relevance:** The project's use of classical ML (Ridge, Random Forest, Gradient Boosting) on engineered IMU features represents an alternative to the deep learning approaches evaluated in RoNIN.
- **Key Findings:** ResNet-based 1D CNN architecture achieved the best overall accuracy for velocity prediction, significantly outperforming IONet.

#### 3.1.4 Liu et al. (2020) — TLIO: Tight Learned Inertial Odometry

- **Citation:** Liu, W., Caruso, D., Ilg, E., Dong, J., Mourikis, A. I., Daniilidis, K., Kumar, V., and Engel, J. (2020). "TLIO: Tight Learned Inertial Odometry." *IEEE Robotics and Automation Letters (RA-L)*, vol. 5, no. 4, pp. 5653–5660.
- **Key Contribution:** Introduced a tightly-coupled framework combining a neural network (for 3D displacement and uncertainty regression from IMU windows) with a stochastic cloning EKF for full 3D pose, velocity, and bias estimation.
- **Relevance:** TLIO is the closest architectural analogue to this project's intended deployment: a learned model producing corrections that are integrated with an EKF.
- **Key Findings:** Achieved low-drift 3D tracking in complex environments using IMU only. The key innovation of outputting uncertainty estimates alongside predictions enabled statistically consistent EKF updates.

### 3.2 AI-Augmented Kalman Filtering

#### 3.2.1 Brossard et al. (2020) — AI-IMU Dead-Reckoning

- **Citation:** Brossard, M., Barrau, A., and Bonnabel, S. (2020). "AI-IMU Dead-Reckoning." *IEEE Transactions on Intelligent Vehicles*, vol. 5, no. 4, pp. 585–595.
- **Key Contribution:** Combined an invariant EKF with recurrent deep neural networks. Instead of predicting navigation states directly, the neural network dynamically adapts the EKF's noise covariance matrices based on incoming inertial data.
- **Relevance:** Demonstrates an elegant hybrid paradigm where ML enhances rather than replaces the model-based filter.
- **Key Findings:** Achieved ~1.10% average translational error on KITTI using IMU only—competitive with methods using LiDAR or stereo vision.

#### 3.2.2 Noureldin et al. (2011) — GPS/INS Integration Utilizing Dynamic Neural Networks

- **Citation:** Noureldin, A., El-Shafie, A., and Bayoumi, M. (2011). "GPS/INS Integration Utilizing Dynamic Neural Networks for Vehicular Navigation." *Information Fusion*, vol. 12, no. 1, pp. 48–59.
- **Key Contribution:** Investigated dynamic neural network architectures (Input-Delayed Neural Networks) for learning time-varying INS error states and providing pseudo-GPS measurements during GPS outages.
- **Relevance:** Directly relevant as an early exemplar of the velocity/position error prediction paradigm—the same conceptual framework as this project.
- **Key Findings:** Dynamic neural networks with temporal embedding significantly outperformed static feedforward networks, reducing position errors during 60-second GPS outages by over 50%.

### 3.3 Learned Inertial Odometry for UAVs

#### 3.3.1 Cioffi et al. (2023) — Learned Inertial Odometry for Autonomous Drone Racing

- **Citation:** Cioffi, G., Bauersfeld, L., Kaufmann, E., and Scaramuzza, D. (2023). "Learned Inertial Odometry for Autonomous Drone Racing." *IEEE Robotics and Automation Letters (RA-L)*, vol. 8, no. 5, pp. 2993–3000.
- **Key Contribution:** Developed a learning-based inertial odometry algorithm specifically designed for agile quadrotor flight.
- **Relevance:** Demonstrates the importance of domain-specific adaptations when applying learned inertial methods to UAVs (vs. pedestrian applications).
- **Key Findings:** Outperformed both traditional VIO systems and generic learned inertial odometry methods during aggressive drone maneuvers.

### 3.4 Data-Driven Zero-Velocity Detection

#### 3.4.1 Wagstaff et al. (2020) — Robust Data-Driven Zero-Velocity Detection

- **Citation:** Wagstaff, B., Peretroukhin, V., and Kelly, J. (2020). "Robust Data-Driven Zero-Velocity Detection for Foot-Mounted Inertial Navigation." *IEEE Sensors Journal*, vol. 20, no. 2, pp. 957–967.
- **Key Contribution:** Replaced traditional fixed-threshold zero-velocity detectors with LSTM-based classifiers for foot-mounted INS.
- **Relevance:** Demonstrates the broader paradigm of using ML to generate pseudo-measurements for Kalman filter correction.
- **Key Findings:** Reduced 3D positioning errors by over 34% compared to standard threshold-based ZUPT methods.

---

## 4. Classical ML for Sensor Data and Time-Series Regression

### 4.1 Esfahani et al. (2019) — AbolDeepIO

- **Citation:** Esfahani, M. A., Wang, H., Wu, K., and Yuan, S. (2019). "AbolDeepIO: A Novel Deep Inertial Odometry Network for Autonomous Vehicles." *IEEE Transactions on Intelligent Transportation Systems*, vol. 21, no. 5, pp. 1941–1950.
- **Key Contribution:** Introduced a triple-channel deep inertial odometry network that explicitly models IMU noise characteristics and sampling time intervals.
- **Relevance:** Demonstrates the importance of noise-aware feature engineering for IMU data—relevant to this project's feature engineering approach.

### 4.2 Tree-Based Ensemble Models for GPS/INS Integration

- **Citation:** *[Removed — the previously cited Sensors 21(13), 4468 (El-Safoury et al., 2021) is an unrelated paper on SO₂ ship emissions monitoring. The intended reference to tree-based ML for GPS/INS integration could not be verified.]*
- **Key Contribution (general literature):** Random Forest and gradient boosting models have been applied to predict INS position and velocity errors during GPS outages in several works, though a specific verified citation with matching DOI is not available for inclusion here.
- **Relevance:** Tree-based ensemble methods are directly analogous to this project's approach—same category of classical ML models for residual prediction.
- **Note:** The result that tree-based methods always outperform simpler baselines should not be assumed.  On this project's test data, zero-correction (doing nothing) may outperform the trained models.

---

## 5. Gaps in Existing Literature

### 5.1 Overemphasis on Deep Learning
The majority of ML-based inertial navigation works employ deep learning architectures. Classical ML models—Ridge Regression, Random Forest, and Gradient Boosting—have been comparatively underexplored, despite advantages in interpretability, training speed, and deployment simplicity.

### 5.2 Velocity-Domain Residual Prediction
Most existing approaches predict absolute velocity/displacement or position/velocity error states. This project's specific formulation—predicting the velocity residual between PX4 EKF2 output and GPS reference—is a novel target definition.

### 5.3 PX4 Ecosystem Integration
Despite PX4 being the most widely deployed open-source UAV autopilot, surprisingly few ML-based navigation correction studies use PX4 flight log data.

### 5.4 Classical ML Baselines for UAV Inertial Navigation
There is a notable absence of systematic comparisons between classical ML models for UAV-specific IMU error prediction.

### 5.5 Computational Efficiency for Edge Deployment
Classical ML models—particularly Ridge Regression and small Random Forest ensembles—can run efficiently on microcontrollers and companion computers typical of UAV platforms.

---

## 6. How This Project Fits In

This project occupies a distinctive position at the intersection of several established research directions:

1. **Interpretability:** Ridge Regression coefficients and Random Forest feature importances provide direct physical insight into which IMU features drive estimator errors.
2. **Practical Deployability:** Computational requirements of Ridge Regression and Random Forest inference are orders of magnitude lower than LSTM or CNN inference, making real-time correction feasible on standard UAV companion computers.
3. **Baseline Establishment:** Establishing strong classical ML baselines before deploying deep architectures determines whether task complexity warrants deep models—a step frequently skipped in the literature.
4. **Target Novelty:** Predicting velocity residuals between PX4 EKF2 and GPS rather than absolute quantities means the model learns to correct the *combined* error of the estimation pipeline.

---

## Summary of Reviewed Works

| # | Authors | Year | Title | Venue | Approach |
|---|---------|------|-------|-------|----------|
| 1 | Groves | 2013 | Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems | Artech House | Navigation theory |
| 2 | Skog and Händel | 2006 | Calibration of a MEMS IMU | IMEKO | Calibration |
| 3 | Noureldin et al. | 2013 | Fundamentals of Inertial Navigation | Springer | AR/GM + ANFIS/ANN |
| 4 | Chen et al. | 2018 | IONet: Learning to Cure the Curse of Drift | AAAI | RNN/LSTM |
| 5 | Yan et al. | 2018 | RIDI: Robust IMU Double Integration | ECCV | Velocity regression |
| 6 | Herath et al. | 2020 | RoNIN: Robust Neural Inertial Navigation | ICRA | CNN/LSTM/TCN |
| 7 | Liu et al. | 2020 | TLIO: Tight Learned Inertial Odometry | IEEE RA-L | NN + EKF |
| 8 | Brossard et al. | 2020 | AI-IMU Dead-Reckoning | IEEE Trans. | RNN + EKF |
| 9 | Noureldin et al. | 2011 | GPS/INS via Dynamic Neural Networks | Information Fusion | Time-delay NN |
| 10 | Cioffi et al. | 2023 | Learned IO for Drone Racing | IEEE RA-L | Model-filter + learning |
| 11 | Wagstaff et al. | 2020 | Data-Driven Zero-Velocity Detection | IEEE Sensors | LSTM + EKF |
| 12 | *(removed — incorrect citation)* | 2021 | Tree-based ML for GPS/INS | — | RF/GBM |

---

## References

1. *(Removed — Sensors 21(13) 4468 is El-Safoury et al., "On-Board Monitoring of SO₂ Ship Emissions," unrelated to GPS/INS.)*
2. Brossard, M., Barrau, A., and Bonnabel, S. (2020). "AI-IMU Dead-Reckoning." *IEEE Transactions on Intelligent Vehicles*, 5(4), 585–595.
3. Chen, C., Lu, X., Markham, A., and Trigoni, N. (2018). "IONet: Learning to Cure the Curse of Drift in Inertial Odometry." In *Proceedings of AAAI*.
4. Cioffi, G., Bauersfeld, L., Kaufmann, E., and Scaramuzza, D. (2023). "Learned Inertial Odometry for Autonomous Drone Racing." *IEEE RA-L*, 8(5), 2993–3000.
5. Esfahani, M. A., Wang, H., Wu, K., and Yuan, S. (2019). "AbolDeepIO: A Novel Deep Inertial Odometry Network." *IEEE Trans. ITS*, 21(5), 1941–1950.
6. Groves, P. D. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems* (2nd ed.). Artech House.
7. Herath, S., Yan, H., and Furukawa, Y. (2020). "RoNIN: Robust Neural Inertial Navigation in the Wild." In *Proceedings of IEEE ICRA*.
8. Liu, W., Caruso, D., Ilg, E., Dong, J., Mourikis, A. I., Daniilidis, K., Kumar, V., and Engel, J. (2020). "TLIO: Tight Learned Inertial Odometry." *IEEE RA-L*, 5(4), 5653–5660.
9. Meier, L., Honegger, D., and Pollefeys, M. (2015). "PX4: A Node-Based Multithreaded Open Source Robotics Framework." In *Proceedings of IEEE ICRA*.
10. Noureldin, A., El-Shafie, A., and Bayoumi, M. (2011). "GPS/INS Integration Utilizing Dynamic Neural Networks." *Information Fusion*, 12(1), 48–59.
11. Noureldin, A., Karamat, T. B., and Georgy, J. (2013). *Fundamentals of Inertial Navigation, Sensor Technologies, and Integration*. Springer.
12. Skog, I. and Händel, P. (2006). "Calibration of a MEMS Inertial Measurement Unit." In *Proceedings of XVII IMEKO World Congress*.
13. Wagstaff, B., Peretroukhin, V., and Kelly, J. (2020). "Robust Data-Driven Zero-Velocity Detection." *IEEE Sensors Journal*, 20(2), 957–967.
14. Yan, H., Shan, Q., and Furukawa, Y. (2018). "RIDI: Robust IMU Double Integration." In *Proceedings of ECCV*.

