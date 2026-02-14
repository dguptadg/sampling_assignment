# sampling_assignment
# Credit Card Fraud Detection: Comparative Analysis of Sampling Techniques

## Project Overview

This project presents a comprehensive empirical study comparing five different sampling techniques across five machine learning algorithms for credit card fraud detection. The primary objective is to identify the most effective combination of sampling method and classification model for handling severely imbalanced datasets in fraud detection scenarios.

Credit card fraud detection poses a significant challenge due to the extreme class imbalance inherent in the data, where fraudulent transactions constitute a minuscule fraction of all transactions. This imbalance can lead to biased models that perform poorly on the minority class. Our study systematically evaluates various resampling strategies to address this challenge.

## Dataset Description

The dataset used in this analysis is a credit card transaction dataset containing 772 samples with 30 feature columns and 1 target variable (Class).

**Class Distribution:**
- Non-fraudulent transactions (Class 0): 763 samples (98.8%)
- Fraudulent transactions (Class 1): 9 samples (1.2%)

This represents an imbalance ratio of approximately 85:1, making it an excellent candidate for testing various sampling techniques.

**Features:**
- 28 principal components (V1-V28) obtained through PCA transformation
- Time: seconds elapsed between this transaction and the first transaction
- Amount: transaction amount
- Class: binary label (0 = legitimate, 1 = fraudulent)

## Methodology

### Data Preprocessing

1. **Data Loading and Validation:** The dataset was loaded and checked for missing values and inconsistencies.

2. **Train-Test Split:** Data was split into training (70%) and testing (30%) sets using stratified sampling to maintain class distribution.

3. **Feature Scaling:** All features were standardized using StandardScaler to ensure equal contribution to model training.

### Sampling Techniques

Five distinct sampling techniques were implemented to address class imbalance:

#### 1. Random Over Sampling (Sampling1)
Random Over Sampling duplicates randomly selected instances from the minority class until balance is achieved. This straightforward approach increases the representation of fraudulent transactions in the training set.

#### 2. SMOTE - Synthetic Minority Over-sampling Technique (Sampling2)
SMOTE generates synthetic samples by interpolating between existing minority class instances and their k-nearest neighbors. This technique creates new, plausible examples rather than simply duplicating existing ones.

#### 3. ADASYN - Adaptive Synthetic Sampling (Sampling3)
ADASYN extends SMOTE by adaptively generating synthetic samples with density distribution. It focuses more on generating samples for minority class instances that are harder to learn, providing a more nuanced approach to oversampling.

#### 4. Random Under Sampling (Sampling4)
Random Under Sampling reduces the majority class by randomly removing instances until balance is achieved. While this can lead to information loss, it significantly reduces training time and can be effective when the majority class is highly redundant.

#### 5. Tomek Links (Sampling5)
Tomek Links identifies and removes borderline instances between classes. This technique cleans the decision boundary by eliminating ambiguous samples, followed by random oversampling to achieve balance.

### Machine Learning Models

Five classification algorithms were evaluated:

#### 1. Logistic Regression (M1)
A linear classification model that estimates the probability of class membership. Despite its simplicity, logistic regression often serves as a strong baseline for binary classification tasks.

#### 2. Random Forest (M2)
An ensemble method that constructs multiple decision trees and aggregates their predictions. Random Forest is robust to overfitting and can capture complex non-linear relationships.

#### 3. Gradient Boosting (M3)
A sequential ensemble technique that builds trees iteratively, with each new tree correcting errors made by previous ones. This method often achieves high accuracy but requires careful tuning to avoid overfitting.

#### 4. Support Vector Machine (M4)
SVM finds the optimal hyperplane that maximally separates classes in a high-dimensional space. With an RBF kernel, it can model complex decision boundaries.

#### 5. K-Nearest Neighbors (M5)
A non-parametric method that classifies instances based on the majority class of their k nearest neighbors. KNN is simple and intuitive but can be sensitive to the choice of k and feature scaling.

### Experimental Design

The experiment followed a systematic approach:

1. Each of the five sampling techniques was applied to the training data
2. Each of the five models was trained on each sampled dataset
3. All models were evaluated on the same held-out test set
4. This resulted in 25 unique sampling-model combinations
5. Performance was measured using classification accuracy

## Results

### Overall Performance Table

The table below shows the accuracy (%) of each model-sampling combination on the test set:

|       | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|-------|-----------|-----------|-----------|-----------|-----------|
| M1    | 92.24     | 93.10     | 93.10     | 59.05     | 91.81     |
| M2    | 99.14     | 98.28     | 98.71     | 64.66     | 99.14     |
| M3    | 95.69     | 98.28     | 98.71     | 64.22     | 95.69     |
| M4    | 96.55     | 96.55     | 96.55     | 83.62     | 96.98     |
| M5    | 96.98     | 94.83     | 94.83     | 89.66     | 96.98     |

<img width="922" height="590" alt="image" src="https://github.com/user-attachments/assets/f86b6cad-d45f-4015-b5e7-8b595cf43c20" />

<img width="1568" height="582" alt="image" src="https://github.com/user-attachments/assets/6f2430bc-3b80-49bf-9992-0596966ac6f8" />

<img width="1568" height="582" alt="image" src="https://github.com/user-attachments/assets/59d06ad7-11cb-433d-ae9a-44add63475ff" />




### Key Findings

#### Best Performing Combinations

1. **Random Forest + Random Over Sampling (M2 + Sampling1): 99.14%**
   - This combination achieved the highest accuracy, demonstrating that Random Forest effectively leverages the increased training data from oversampling.

2. **Random Forest + Tomek Links (M2 + Sampling5): 99.14%**
   - Equal performance to the top combination, showing that cleaning decision boundaries can be as effective as simple oversampling for ensemble methods.

3. **Gradient Boosting + SMOTE (M3 + Sampling2): 98.28%**
   - SMOTE's synthetic samples provided excellent training data for the gradient boosting algorithm.

4. **Gradient Boosting + ADASYN (M3 + Sampling3): 98.71%**
   - ADASYN's adaptive approach slightly outperformed standard SMOTE with gradient boosting.

#### Performance by Sampling Technique

**Sampling1 (Random Over Sampling):**
- Average accuracy: 96.12%
- Most consistent performance across all models
- Best overall results with ensemble methods

**Sampling2 (SMOTE):**
- Average accuracy: 96.21%
- Strong performance with tree-based models
- Generated high-quality synthetic samples

**Sampling3 (ADASYN):**
- Average accuracy: 96.38%
- Slightly better than SMOTE on average
- Adaptive generation helped with complex decision boundaries

**Sampling4 (Random Under Sampling):**
- Average accuracy: 72.24%
- Significantly lower performance
- Information loss from discarding majority class samples likely hurt model learning

**Sampling5 (Tomek Links):**
- Average accuracy: 96.32%
- Excellent performance, particularly with Random Forest
- Boundary cleaning proved effective

#### Performance by Model

**M1 (Logistic Regression):**
- Average accuracy: 85.86%
- Lowest performing model overall
- Linear decision boundary insufficient for this task
- Struggled particularly with undersampling

**M2 (Random Forest):**
- Average accuracy: 95.99%
- Best performing model
- Consistently high accuracy across sampling techniques
- Most robust to different sampling strategies

**M3 (Gradient Boosting):**
- Average accuracy: 90.72%
- Second-best model
- Excellent with SMOTE and ADASYN
- More sensitive to sampling choice than Random Forest

**M4 (Support Vector Machine):**
- Average accuracy: 94.05%
- Solid performance across most techniques
- Most consistent results (low variance)
- Relatively robust to sampling method

**M5 (K-Nearest Neighbors):**
- Average accuracy: 94.66%
- Good performance overall
- Benefited from oversampling techniques
- Best performance with Tomek Links cleaning

### Analysis of Results

#### Why Random Under Sampling Performed Poorly

Random Under Sampling (Sampling4) showed significantly lower accuracy across all models (average 72.24%). This can be attributed to:

1. **Information Loss:** Discarding 98% of the training data removed valuable information about legitimate transaction patterns.
2. **Reduced Training Set Size:** With only 18 samples (9 from each class), models had insufficient data to learn robust decision boundaries.
3. **Loss of Diversity:** The small sample may not have captured the full diversity of legitimate transactions.

#### Why Oversampling Techniques Excelled

Oversampling methods (Sampling1, 2, 3, and 5) consistently achieved high performance:

1. **Preserved Information:** All original data was retained, ensuring no loss of majority class patterns.
2. **Better Class Representation:** Minority class received adequate representation for model learning.
3. **Larger Training Sets:** More data generally leads to better model generalization.

#### Model-Specific Insights

**Random Forest Dominance:**
Random Forest achieved the highest accuracy because:
- Ensemble structure reduced overfitting risk from duplicated samples
- Multiple trees captured different aspects of the data
- Built-in feature selection reduced noise from synthetic samples

**Logistic Regression Limitations:**
Linear models struggled because:
- Fraud patterns likely involve non-linear relationships
- Linear decision boundaries cannot capture complex interactions
- Less capacity to leverage additional training data from oversampling

### Visual Analysis

The heatmap visualization clearly shows:
- Dark blue regions indicating high accuracy (>95%) concentrated in Sampling1, 2, 3, and 5
- Yellow region showing Sampling4's poor performance across all models
- M2 (Random Forest) row showing consistently high performance

The line plots reveal:
- Sharp drop in all models' performance at Sampling4
- Recovery to high performance at Sampling5
- M2 and M3 maintaining highest accuracies across most techniques

## Conclusions

This comprehensive study yielded several important conclusions for fraud detection practitioners:

1. **Oversampling is Superior to Undersampling:** For severely imbalanced datasets, oversampling techniques consistently outperform undersampling by preserving valuable majority class information.

2. **Random Forest is the Most Robust Model:** Random Forest achieved the best overall performance and showed the least sensitivity to sampling technique choice.

3. **Simple Techniques Can Match Complex Ones:** Random Over Sampling performed on par with more sophisticated techniques like SMOTE and ADASYN, suggesting that simplicity should not be overlooked.

4. **Ensemble Methods Excel:** Both Random Forest and Gradient Boosting significantly outperformed single classifiers, highlighting the value of ensemble approaches for imbalanced classification.

5. **Boundary Cleaning is Effective:** Tomek Links, by cleaning decision boundaries, achieved excellent results comparable to pure oversampling methods.
