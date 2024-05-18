# Fraud-Transaction-Detection
INSAID - Data Science &amp; Machine Learning Internship 

**INTRODUCTION**

This project focuses on building and evaluating machine learning models to detect fraudulent transactions. Using a highly imbalanced dataset where fraudulent transactions are significantly fewer than legitimate ones, we implemented and compared two classifiers: Decision Tree and Random Forest. Our aim was to achieve high precision and recall for detecting fraud, minimizing the risk of false positives and false negatives.

**DATA**

The dataset comprises various features related to transactions, which help identify potential fraud. Before model training, we performed data preprocessing, which included handling null values and encoding categorical data.

**MODELS** **USED**

**_Decision_** **_Tree_**

A Decision Tree is a simple yet powerful model that makes decisions based on feature values. It is known for its interpretability and effectiveness in handling imbalanced datasets by making decisions in a hierarchical manner.

**_Random_** **_Forest_**

A Random Forest is an ensemble model that combines multiple decision trees to improve overall performance. By aggregating the decisions from multiple trees, it reduces the risk of overfitting and enhances the model's ability to generalize to new data.

**EVALUATION** **METRICS**

We used several metrics to evaluate model performance:

_Accuracy_: The proportion of correctly classified transactions.

_Precision_: The proportion of predicted fraudulent transactions that are actually fraudulent.

_Recall_: The proportion of actual fraudulent transactions that are correctly identified.

_F1_-_Score_: The harmonic mean of precision and recall, providing a balance between the two.

_Area_ _Under_ _the_ _Curve_ (AUC): A metric that considers the true positive rate and false positive rate across different threshold settings.

**RESULTS**

**_Decision_** **_Tree_**

_Accuracy_: 99.63%

_Precision_: 0.42 (for fraud detection)

_Recall_: 0.41 (for fraud detection)

_F1_-_Score_: 0.42 (for fraud detection)

_AUC_: 0.70

__Confusion__ __Matrix__:

True Positives (TP): 11

False Positives (FP): 15

True Negatives (TN): 8447

False Negatives (FN): 16

**_Random_** **_Forest_**

_Accuracy_: 99.79%

_Precision_: 1.00 (for fraud detection)

_Recall_: 0.33 (for fraud detection)

_F1_-_Score_: 0.50 (for fraud detection)

_AUC_: 0.67

__Confusion__ __Matrix__:

True Positives (TP): 9

False Positives (FP): 0

True Negatives (TN): 8462

False Negatives (FN): 18

**CONCLUSION**

Both models achieved high accuracy. The Decision Tree model offers a balanced trade-off between precision and recall, making it more reliable for identifying fraudulent transactions. The Random Forest model achieves perfect precision, ensuring that legitimate transactions are not incorrectly flagged as fraudulent, though it has lower recall. Combining the strengths of both models could provide a robust solution for detecting fraud.

**REQUIREMENTS**

To run this project, the following Python libraries are required:

pandas

numpy

scikit-learn

matplotlib

seaborn

Install these libraries using pip:
pip install pandas numpy scikit-learn matplotlib seaborn

**USAGE**

_Data_ _Preprocessing_: Load the dataset and preprocess it by handling null values and encoding categorical variables.

_Model_ _Training_: Train both Decision Tree and Random Forest models on the preprocessed data.

_Model_ _Evaluation_: Evaluate the models using the defined metrics and compare their performance.

_Visualization_: Plot confusion matrices and other visualizations to understand model performance.

This project demonstrates the application of machine learning techniques to detect fraudulent transactions, emphasizing the importance of precision and recall in fraud detection tasks. By carefully evaluating and comparing models, we can choose the best approach to minimize financial risks associated with fraud.
