# Team-Project-Group4
Data Science Challenge: H1N1 Vaccine Prediction
This project is a submission for the Assignment 1 - Group project - Data Science Challenge. The objective is to analyze a dataset from a survey on H1N1 vaccine opinions and behaviors and to build a machine learning model to predict the likelihood of a respondent receiving the H1N1 vaccine.


This repository contains the full data science pipeline, from initial exploration to final model submission, as required by the assignment brief.

Project Structure
Data_Science_Project (1).ipynb: The main Jupyter Notebook containing all code for analysis, preprocessing, modeling, and submission.

dataset_B_training.csv: The raw training data provided for the challenge.

dataset_B_testing (1).csv: The raw testing data provided for the challenge.

submission_*.csv: The 5 final prediction files for submission, ordered by perceived performance.

eda_*.png: Saved plots from the Exploratory Data Analysis.

README.md: This file.

Methodology
The project follows a standard data science pipeline as outlined in the assignment brief.





1. Data Exploration (EDA)
Before any preprocessing, a deep exploratory data analysis was performed on the raw training data to understand its structure, find patterns, and inform our modeling decisions.

Target Variable Analysis: Plotted the distribution of the h1n1_vaccine target variable. This revealed a significant class imbalance (approx. 79% No, 21% Yes), confirming that AUC-ROC would be a more appropriate success metric than simple accuracy.

Missing Data: A heatmap was generated to visualize missing values. This clearly showed that health_insurance and employment_sector had a very high percentage of missing data, justifying their removal during preprocessing.

Key Feature Analysis: Created bivariate plots for key features against the target. This revealed strong predictive signals from:

doctor_recc_h1n1: A doctor's recommendation was highly correlated with receiving the vaccine.

opinion_h1n1_vacc_effective: Belief in the vaccine's effectiveness was a strong predictor.

opinion_h1n1_risk: Higher perceived risk of contracting H1N1 correlated with a higher vaccination rate.

Correlation: A heatmap of all numeric/ordinal features was generated to check for multicollinearity and identify other relationships.

2. Data Pre-processing
A ColumnTransformer and Pipeline from scikit-learn were used to create a robust and repeatable preprocessing workflow that prevents data leakage.

Target/ID Separation: The h1n1_vaccine target variable and respondent_id were separated from the feature set.

Column Dropping: health_insurance and employment_sector were dropped due to high missingness.


Imputation:

Numeric Features: Missing values were imputed using the median.

Categorical Features: Missing values were imputed using the most_frequent value (mode).


Encoding & Scaling:

Categorical: OneHotEncoder was applied to all object-type columns to convert them into a machine-readable format.

Numeric: StandardScaler was applied to all numeric/ordinal features to normalize their distributions, which is important for models like Logistic Regression.

3. Modelling & Performance Analysis

Metric Selection: AUC-ROC was used as the primary evaluation metric to handle the class imbalance, as required for an appropriate metric.


Baseline Models: The processed data was split into a stratified 80/20 train/validation set to get a robust performance measure on unseen data. Three models were compared:


LogisticRegression (with class_weight='balanced')

RandomForestClassifier (with class_weight='balanced')

GradientBoostingClassifier

Performance: Both LogisticRegression and GradientBoostingClassifier showed strong baseline performance (AUC > 0.81), while RandomForestClassifier was slightly behind.


Hyper-parameter Tuning: RandomizedSearchCV (with 5-fold cross-validation) was used to find the best parameters for the top two models: LogisticRegression and GradientBoostingClassifier. This search successfully improved the GradientBoostingClassifier's performance to an AUC of ~0.8197, making it our top model.

4. Submission
Based on the tuning and validation results, the 5 best models were selected and re-trained on the entire processed training dataset. These models were then used to predict probabilities on the processed test set.

The 5 submission files are ordered from best to worst, as required:


submission_1_tuned_gb.csv: (Best Model) The tuned Gradient Boosting model.

submission_2_tuned_lr.csv: The tuned Logistic Regression model.

submission_3_base_gb.csv: The baseline (default) Gradient Boosting model.

submission_4_base_lr.csv: The baseline (default) Logistic Regression model.

submission_5_base_rf.csv: The baseline (default) Random Forest model.

How to Run
Ensure you have Python 3.x and the required libraries installed (see Technology Stack).

Clone this repository.

Place the raw data files (dataset_B_training.csv and dataset_B_testing (1).csv) in the root directory.

Open and run the Data_Science_Project (1).ipynb notebook from top to bottom.

The notebook is self-contained and will:

Generate and save (and display) the EDA plots.

Perform all data preprocessing.

Train, tune, and evaluate all models.

Save the 5 final submission_*.csv files to the root directory.

Technology Stack
Python 3

Jupyter

pandas: For data loading and manipulation.

numpy: For numerical operations.

scikit-learn: For preprocessing, pipelines, modeling, and metrics.

matplotlib & seaborn: For data visualization.

Team Members
[List team members here]
