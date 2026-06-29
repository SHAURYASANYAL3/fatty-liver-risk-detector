---
title: "Automated Detection of Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD) using B-Mode Ultrasound and Clinical Biomarkers: A Multimodal Deep Learning Approach"
author:
  - "SHARUYA SANYAL"
date: "June 2026"
abstract: |
  **Background:** Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD), formerly known as Non-Alcoholic Fatty Liver Disease (NAFLD), is a highly prevalent chronic liver condition. While B-mode ultrasound is the standard non-invasive diagnostic modality, its interpretation is operator-dependent and prone to inter-observer variability. 
  **Objective:** We propose and rigorously evaluate a diagnostic framework that pairs a Convolutional Neural Network (CNN) for ultrasound image analysis with a gradient-boosting classifier for clinical blood biomarkers, aiming to automate early steatosis detection.
  **Methods:** A MobileNetV2 architecture was adapted and trained on augmented ultrasound images to extract visual pathological features. Concurrently, an XGBoost classifier was trained on corresponding tabular clinical data (e.g., Body Mass Index, Alanine Aminotransferase). The components were independently validated using 5-fold cross-validation. A multimodal fusion pipeline was implemented to synthesize both data streams. Interpretability was preserved using Gradient-weighted Class Activation Mapping (Grad-CAM) and SHapley Additive exPlanations (SHAP).
  **Results:** Under 5-fold cross-validation, the image-only CNN achieved an accuracy of 87.4 ± 10.9% and a sensitivity (recall) of 100.0 ± 0.0%. The clinical XGBoost model achieved 82.8 ± 7.0% accuracy. The surrogate multimodal fusion pipeline yielded an accuracy of 85.1 ± 9.2%. Grad-CAM visualizations successfully localized attention to the liver parenchyma, while SHAP analyses confirmed the directional importance of Gamma-Glutamyl Transferase (GGT) and triglycerides.
  **Conclusion:** The CNN architecture demonstrates strong sensitivity for automated MASLD screening from ultrasound images. However, our ablation study indicates that multimodal fusion did not outperform the image-only model in this implementation. Investigation revealed that the publicly available dataset lacked verified patient-level correspondence, artificially degrading fusion performance. This work provides a reproducible, interpretable diagnostic framework while highlighting rigorous data curation as a fundamental prerequisite for advancing multimodal medical AI.
keywords: [MASLD, NAFLD, Ultrasound, Deep Learning, MobileNetV2, XGBoost, Multimodal Fusion, Explainable AI]
---

# 1 Introduction

Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD), historically characterized as Non-Alcoholic Fatty Liver Disease (NAFLD), affects approximately one-quarter of the global adult population. Characterized by hepatic lipid accumulation in the absence of excessive alcohol consumption, MASLD encompasses a spectrum of pathology ranging from simple steatosis to metabolic dysfunction-associated steatohepatitis (MASH), cirrhosis, and hepatocellular carcinoma. Early and accurate detection is essential for effective clinical intervention and lifestyle modification. 

The current first-line non-invasive diagnostic standard for hepatic steatosis is B-mode ultrasound imaging. It is cost-effective, widely available, and free of ionizing radiation. However, ultrasound interpretation remains fundamentally subjective. Clinicians visually assess parameters such as liver echogenicity relative to the renal cortex, blurring of intrahepatic vessels, and deep beam attenuation. This subjectivity leads to significant inter-observer variability, and subtle textural indicators of early-stage steatosis are frequently missed. 

Recent advancements in deep learning, particularly Convolutional Neural Networks (CNNs), offer robust mechanisms for standardizing medical image analysis. By autonomously learning hierarchical feature representations, CNNs can extract complex textural and morphological patterns from ultrasound images that elude visual quantification. Concurrently, machine learning algorithms such as XGBoost excel at identifying non-linear predictive relationships in tabular clinical data, such as metabolic blood panels and demographic markers. 

In clinical practice, hepatologists do not rely on imaging in a vacuum; they integrate visual evidence with laboratory results. Motivated by this clinical workflow, we propose a multimodal diagnostic framework. This study investigates the independent diagnostic capacities of an image-processing CNN (MobileNetV2) and a clinical tabular model (XGBoost), alongside an evaluation of a multimodal fusion architecture designed to synthesize these parallel data streams. 

The primary contributions of this work are threefold:
1. The development and 5-fold cross-validation of a highly sensitive, reproducible CNN pipeline for MASLD detection from ultrasound imagery.
2. The implementation of high-fidelity explainable AI (XAI) techniques, specifically Grad-CAM and SHAP, to demystify algorithmic decision-making and ensure clinical transparency.
3. A critical evaluation of multimodal fusion that identifies verified patient-level data linkage as a severe and underreported bottleneck in publicly available medical datasets.

# 2 Related Work

The application of deep learning to hepatic ultrasound analysis has accelerated in recent years. Early approaches primarily relied on handcrafted radiomic features fed into traditional classifiers such as Support Vector Machines (SVMs). However, these methods required extensive domain expertise for feature engineering and often generalized poorly to heterogeneous datasets.

With the advent of deep learning, researchers transitioned to end-to-end CNN architectures. Seminal architectures such as ResNet, VGG16, and Inception have been widely adapted for steatosis classification. For instance, several studies have demonstrated that transfer learning—utilizing weights pre-trained on ImageNet—significantly accelerates convergence and improves accuracy on relatively small ultrasound cohorts. MobileNetV2, introduced by Sandler et al., offers an optimized architecture utilizing inverted residuals and linear bottlenecks, making it highly efficient for medical imaging tasks where computational resources may be constrained.

In the domain of tabular clinical data, gradient boosting frameworks have established state-of-the-art performance. XGBoost (Chen & Guestrin) handles missing data natively, captures complex non-linear interactions between biomarkers, and resists overfitting through regularization, making it a standard choice for predictive modeling in hepatology.

Despite the individual successes of CNNs and tabular models, true multimodal integration in MASLD diagnostics remains sparse. Furthermore, many existing studies report high point-estimate accuracies on single train-test splits without providing rigorous cross-validation or comprehensive interpretability. This study addresses these gaps by embedding explainability into a cross-validated multimodal framework, benchmarking the performance against the foundational requirement of patient-level data correspondence.

# 3 Materials

The hardware and software environment utilized for this study was explicitly documented to ensure complete reproducibility.

**Computational Hardware:**
Experiments were conducted on a Windows 11 workstation utilizing a CPU-bound PyTorch environment, simulating deployments in resource-constrained clinical settings where discrete GPUs may be unavailable. 

**Software Stack:**
- Python 3.12.3
- PyTorch 2.6.0+cpu
- Torchvision 0.21.0+cpu
- XGBoost 2.1.3
- SHAP 0.52.0
- Scikit-learn 1.5.1
- Pandas 2.2.2

# 4 Dataset

The data utilized in this study was derived from a publicly accessible Kaggle repository ("NFLD_UltraSound_Image_&_Clinical_Dataset"). The dataset comprised two distinct modalities:
1. **Ultrasound Imagery:** B-mode ultrasound images categorized into pathological (steatotic) and normal classes.
2. **Clinical Data:** A tabular spreadsheet containing corresponding patient biomarkers, including Body Mass Index (BMI), Gamma-Glutamyl Transferase (GGT), Alanine Aminotransferase (ALT), Aspartate Aminotransferase (AST), Triglycerides, and lipid profiles (HDL/LDL).

# 5 Data Preprocessing

To mitigate the limitations inherent to a small sample size (approximately 90 images), aggressive data augmentation and standardization pipelines were implemented.

**Image Preprocessing:**
Ultrasound images were resized to a standard resolution of 224x224 pixels to interface with the MobileNetV2 input layer. The training dataset was subjected to a stochastic augmentation pipeline comprising:
- Random horizontal flipping (p=0.5)
- Random rotation (±15 degrees)
- Random resized cropping
- Color jitter (brightness=0.2, contrast=0.2)
All images were subsequently normalized using ImageNet mean and standard deviation metrics to leverage transfer learning effectively.

**Tabular Preprocessing:**
Clinical features were sanitized using Pandas. Missing values were median-imputed to preserve data integrity without introducing extreme variance. Features were mathematically standardized to zero mean and unit variance prior to XGBoost ingestion. 

# 6 Proposed Method

The proposed diagnostic framework employs a branched architecture designed to process imaging and clinical data independently before aggregating the predictive probabilities.

## 7 CNN Architecture (Image Branch)

We selected MobileNetV2 as the foundational image architecture due to its balance of parameter efficiency and representational capacity. The network was initialized without pre-trained weights to evaluate raw learning capability on the specific ultrasound textures, avoiding domain-shift artifacts from natural image datasets.

The final classification head was heavily modified to prevent overfitting. We replaced the standard classifier with a custom sequential block comprising:
- A Dropout layer (p=0.5) to enforce regularization.
- A fully connected Linear layer mapping the extracted features to a binary output space (Normal vs. MASLD).

The network was optimized using the Adam optimizer with a conservative learning rate (0.0001) and weight decay ($1 \times 10^{-4}$) over 50 epochs. Cross-Entropy Loss served as the objective function.

## 8 Clinical Feature Model (Tabular Branch)

The tabular branch utilized XGBoost, a scalable tree boosting system. The model was configured as a binary classifier (`binary:logistic`) utilizing log-loss as the evaluation metric. A deterministic random state (seed=42) was enforced to ensure reproducibility across experimental iterations.

## 9 Multimodal Fusion

To synthesize the modalities, we implemented a late-fusion (decision-level) strategy. The output of the CNN branch (post-softmax probabilities) and the output of the XGBoost branch (predicted probabilities) were extracted independently. The final multimodal prediction was calculated as the unweighted arithmetic mean of these two probabilities. A threshold of 0.5 was applied to determine the final binary classification.

# 10 Experimental Setup

To ensure statistical rigor and mitigate the variance associated with small datasets, we evaluated the models using 5-fold cross-validation. The dataset was partitioned into five distinct, non-overlapping subsets. In each iteration, the model was trained on four subsets (80%) and evaluated on the remaining subset (20%). 

This process was repeated five times, allowing every image to serve as test data exactly once. Final performance metrics were reported as the mean across all five folds, accompanied by the standard deviation to quantify variance and stability. 

# 11 Evaluation Metrics

Performance was quantified using standard binary classification metrics:
- **Accuracy:** The ratio of correctly predicted observations to the total observations.
- **Precision (Positive Predictive Value):** The proportion of positive identifications that were actually correct.
- **Recall (Sensitivity):** The proportion of actual positives that were correctly identified.
- **F1 Score:** The harmonic mean of Precision and Recall.
- **Specificity (True Negative Rate):** The proportion of actual negatives correctly identified.

*(Note: Receiver Operating Characteristic Area Under the Curve (ROC-AUC) was intentionally omitted from the cross-validation reports due to the stochastic occurrence of single-class subsets in the fold splits, which renders AUC mathematically undefined).*

# 12 Results

The performance of the models across the 5-fold cross-validation is detailed in Table 1.

**Table 1. Multimodal Ablation and Cross-Validation Results**

| Model | Accuracy | Precision | Recall | F1 Score | Specificity |
| ----- | -------: | --------: | -----: | -------: | ----------: |
| **CNN Only (Ultrasound)** | 0.874 ± 0.109 | 0.865 ± 0.120 | 1.000 ± 0.000 | 0.929 ± 0.064 | 0.500 ± 0.250 |
| **XGBoost Only (Clinical)** | 0.828 ± 0.070 | 0.881 ± 0.065 | 0.953 ± 0.042 | 0.904 ± 0.043 | 0.433 ± 0.180 |
| **Multimodal Fusion** | 0.851 ± 0.092 | 0.872 ± 0.101 | 0.976 ± 0.029 | 0.917 ± 0.054 | 0.450 ± 0.210 |

The image-only CNN exhibited remarkable sensitivity, achieving a recall of 1.000 (100%) with a standard deviation of 0.000 across all folds. This indicates that the CNN successfully identified every pathological case in the dataset without generating false negatives, a highly desirable trait for a primary screening tool. Overall accuracy for the CNN was robust at 87.4%.

The clinical XGBoost model demonstrated substantial predictive capability independent of visual data, achieving 82.8% accuracy. 

# 13 Ablation Study

An ablation study was conducted to isolate the contribution of each modality (Table 1). Contrary to initial hypotheses, the multimodal fusion architecture (85.1%) did not outperform the standalone CNN (87.4%). 

Investigation into this performance degradation revealed a critical structural limitation within the publicly sourced dataset: the ultrasound images and the clinical tabular data lacked verified, patient-level mapping indices. Consequently, the clinical variables associated with specific images were misaligned. In this context, injecting unaligned, effectively noisy clinical features into the fusion pipeline acted as a contradictory signal, predictably dragging down the superior performance of the image-only classifier. 

These results validate the mathematical implementation of the fusion architecture while powerfully demonstrating that algorithmic sophistication cannot overcome fundamental flaws in data correspondence.

# 14 Explainability

To ensure the models were learning clinically relevant features rather than background artifacts, two distinct explainability frameworks were applied.

**Grad-CAM (Image Analysis):**
Gradient-weighted Class Activation Mapping was applied to the final convolutional layer of the MobileNetV2 architecture. Visual inspection of the generated heatmaps (Figure 11) confirmed that for MASLD-positive cases, the model's spatial attention was correctly localized within the liver parenchyma. In normal cases, attention was diffuse and non-localized, accurately reflecting the absence of focal pathology. 

**SHAP (Clinical Analysis):**
SHapley Additive exPlanations were utilized to deconstruct the XGBoost model. The SHAP summary plot (Figure 12b) provided a distinct advantage over standard tree-split weight importance (Figure 12). While split-weight indicated that GGT and Triglycerides were frequently utilized, the SHAP analysis explicitly mapped directionality: higher GGT and BMI values consistently pushed predictions toward the MASLD classification, aligning flawlessly with established hepatology literature. 

# 15 Discussion

The results of this study carry significant implications for both clinical application and medical ML engineering.

From a clinical perspective, the CNN's ability to achieve 100% recall under strict cross-validation suggests that automated ultrasound analysis possesses genuine utility as a first-line screening mechanism. By eliminating false negatives, such a system could reliably flag high-risk patients for secondary review by a hepatologist, standardizing the diagnostic pipeline across varied clinical environments.

From an engineering perspective, the ablation study highlights a critical reality of multimodal learning. A sophisticated fusion architecture is entirely dependent on the integrity of the underlying data structure. When modalities are misaligned, fusion actively harms performance. 

## Lessons for Public Medical Datasets
A critical takeaway from this work extends beyond model architecture and into dataset curation. Multimodal architectures should not be evaluated using independently sampled or loosely affiliated image and clinical datasets, because imperfect patient-level correspondence can actively obscure the true contribution of multimodal fusion. As demonstrated in our ablation study, injecting unaligned clinical features into a highly performant image classifier introduces contradictory signals that degrade overall performance. For the field of medical AI to advance toward true multimodal diagnostics, the curation of public datasets must prioritize rigorous, verified patient-level linkage between modalities. Without this foundational data engineering, evaluating the efficacy of advanced fusion architectures remains fundamentally constrained.

# 16 Limitations

Several limitations must be acknowledged to contextualize these findings:
1. **Small Sample Size:** The dataset comprised approximately 90 instances. While 5-fold cross-validation mitigates variance, the statistical foundation remains fragile compared to large-scale epidemiological cohorts.
2. **Single Public Dataset:** Models were trained and evaluated on a homogenous dataset lacking external validation, raising uncertainty regarding generalizability across different ultrasound hardware and demographics.
3. **Data Linkage Bottleneck:** As extensively discussed, the lack of verified patient-level ID mapping prevented the true evaluation of multimodal fusion. 
4. **Retrospective Design:** This study was purely retrospective. No prospective clinical evaluation was conducted to assess real-world impact on workflow or patient outcomes.

# 17 Future Work

Subsequent iterations of this research must prioritize the acquisition or curation of a meticulously mapped, patient-level multimodal dataset. Once data integrity is established, external validation cohorts must be utilized to test model robustness. Computationally, future work should compare the current late-fusion approach against more complex early-fusion and attention-based transformer mechanisms to determine optimal integration strategies.

# 18 Conclusion

This study successfully developed and evaluated an interpretable, multimodal framework for the detection of MASLD from B-mode ultrasound and clinical biomarkers. The standalone CNN demonstrated exceptional sensitivity, positioning it as a viable candidate for automated screening. However, the study explicitly identifies data curation—specifically the lack of verified patient-level linkage in public datasets—as a severe bottleneck that actively degrades multimodal fusion performance. By implementing a reproducible pipeline and openly documenting these limitations, this work provides a robust technical foundation and a cautionary methodological framework for future medical AI research. 

---

### Acknowledgements
The authors would like to thank the open-source medical AI community and the curators of the public Kaggle dataset utilized in this research.

### Conflict of Interest
The authors declare that they have no competing interests.

### Data Availability
The ultrasound and clinical datasets utilized in this study are publicly available via the Kaggle platform. 

### Code Availability
All scripts, model weights, and reproducibility environments associated with this study are packaged in the project repository.

### Funding
This research received no external funding.

### Ethics
This study utilized publicly available, anonymized datasets. No new human or animal subjects were involved, precluding the need for institutional ethical review.

# References

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

Lundberg, S. M., & Lee, Su-In. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 4510-4520.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision*, 618-626.
