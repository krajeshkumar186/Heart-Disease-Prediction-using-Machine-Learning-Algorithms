# Heart-Disease-Prediction-using-Machine-Learning Algorithms 
Heart disease is a leading cause of death worldwide, and timely prediction can play a key role in preventing it. Leveraging data-driven systems for predicting heart disease can transform how medical professionals and researchers approach prevention and treatment. Machine Learning (ML) offers powerful tools for predicting heart disease based on patient data, providing accurate predictions that can assist in early diagnosis, enabling timely intervention, and improving patient outcomes.

Project Overview
This project focuses on the development of a heart disease prediction model using several ML algorithms. The model was built by analyzing patient data, which was carefully processed to ensure high-quality input. Various supervised learning techniques were employed to train the model, and predictions were made based on key health parameters.

Key Steps in the Process:
Data Preprocessing:

Data Cleaning: Handling missing values and outliers.
Feature Scaling: Standardization/Normalization to scale features like age, blood pressure, cholesterol levels, etc.
Splitting Data: Dividing the dataset into training and testing sets (commonly in an 80:20 ratio) to validate the model's performance.
Feature Selection:

Identifying significant health features influencing heart disease such as:
Age
Gender
Chest pain type
Resting blood pressure
Serum cholesterol
Fasting blood sugar
Resting electrocardiographic results
Maximum heart rate achieved
Exercise-induced angina
ST depression
Slope of peak exercise ST segment
Number of major vessels colored by fluoroscopy
Thalassemia
Model Training:

A wide variety of ML algorithms were applied to build classification models predicting the likelihood of heart disease. These models were trained using Python libraries like Scikit-learn and Keras, which provided tools for building, training, and evaluating the models.
Model Evaluation:

Accuracy was used as the key evaluation metric, although other metrics such as precision, recall, F1 score, and ROC-AUC curve were also considered to ensure a robust assessment of model performance.
Machine Learning Algorithms Used
1. Logistic Regression (Scikit-learn):
Logistic regression is a simple yet effective algorithm for binary classification tasks. It calculates the probability of the target class (heart disease) and applies a threshold to predict the class. It is well-suited for problems where the target variable is binary, as in this case.

2. Naive Bayes (Scikit-learn):
Naive Bayes is based on the Bayes theorem, assuming independence between features. Despite its simplicity, it can be effective for classification tasks, especially when the dataset's feature independence assumption holds.

3. Support Vector Machine (SVM - Linear) (Scikit-learn):
SVM tries to find the optimal hyperplane that separates the data into different classes. For this project, a linear kernel was used, as heart disease prediction may exhibit a linear relationship between features.

4. K-Nearest Neighbors (KNN - Scikit-learn):
KNN is a non-parametric algorithm that classifies data points based on the majority class among its k-nearest neighbors. It is simple and effective, especially when the dataset is not too large.

5. Decision Tree (Scikit-learn):
Decision Tree is a non-linear algorithm that recursively splits the data based on features that maximize the information gain. It is intuitive and interpretable, making it easier to understand why certain predictions were made.

6. Random Forest (Scikit-learn):
Random Forest is an ensemble method combining multiple decision trees to improve accuracy and reduce overfitting. By averaging the results of multiple trees, it increases the robustness of predictions. This model achieved the highest accuracy of 95% in the project.

7. XGBoost (Scikit-learn):
XGBoost is an advanced ensemble learning technique that uses gradient boosting to build strong classifiers. It improves prediction accuracy by iteratively adding weak learners (decision trees) and correcting previous errors.

8. Artificial Neural Network (ANN - Keras):
An ANN was implemented with one hidden layer. Neural networks are particularly powerful for complex datasets as they can capture intricate patterns in the data. Keras was used for building and training the network. Despite the small size of the dataset, ANN provided decent predictive performance.

Results
The Random Forest algorithm provided the highest accuracy of 95%, outperforming other models. However, other models like XGBoost, Decision Trees, and Logistic Regression also showed competitive performance. The choice of the best algorithm depends on factors like interpretability, computational cost, and deployment requirements.

Algorithm	Accuracy:
Logistic Regression	85%
Naive Bayes	83%
Support Vector Machine	84%
K-Nearest Neighbors	87%
Decision Tree	90%
Random Forest	95%
XGBoost	92%
Artificial Neural Network	88%

Dataset Used
The dataset used for this project is the UCI Heart Disease dataset, which contains 303 observations with 14 features representing various health metrics. The dataset can be found on Kaggle:- https://www.kaggle.com/datasets/ronitf/heart-disease-uci

Conclusion
This project demonstrates how Machine Learning algorithms can be effectively used to predict the likelihood of heart disease. Random Forest emerged as the most accurate model, achieving 95% accuracy, but other models also performed well. The combination of multiple algorithms allows flexibility in prediction depending on the trade-offs between accuracy, interpretability, and computational efficiency. This model could serve as a foundation for building more advanced predictive systems in healthcare, aiding doctors and patients in early diagnosis and better management of heart diseases.
