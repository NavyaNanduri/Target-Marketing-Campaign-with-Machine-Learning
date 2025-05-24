# Target Marketing Campaign with Machine Learning

## Project Objective

This project aims to predict whether banking customers will purchase a variable annuity product using various machine learning models. The insights gained from this analysis can be used to optimize target marketing campaigns.

## Data

The analysis is based on a SAS dataset (`develop (1).sas7bdat`). The dataset contains information about banking customers.

## Methodology

The project follows a standard machine learning workflow:

1.  **Data Loading and Initial Inspection:**
    *   Load the SAS dataset using pandas.
    *   Perform basic checks on the data shape, data types, and summary statistics.
    *   Analyze the distribution of the target variable (`Ins`).

2.  **Exploratory Data Analysis (EDA):**
    *   Identify and handle missing values and potential outliers.
    *   Visualize the distribution of key numerical features (e.g., `Income`, `Age`).
    *   Analyze the distribution of categorical variables.
    *   Visualize relationships between variables using techniques like correlation heatmaps and box plots.

3.  **Data Preprocessing:**
    *   Handle missing values (if any).
    *   Encode nominal categorical variables using one-hot encoding.
    *   Scale numerical features using `StandardScaler`.
    *   Split the data into training and testing sets (50/50 split) using stratification to maintain the target variable distribution.

4.  **Model Building and Evaluation:**
    *   Train and evaluate the following machine learning models:
        *   **Decision Tree:** Built with a specified maximum depth.
        *   **Logistic Regression:** Trained on scaled data, with analysis of coefficients and odds ratios.
        *   **Neural Network:** Implemented using Keras/TensorFlow with different batch sizes for comparison.
        *   **Random Forest:** Trained with a specific number of estimators, with feature importance analysis.
        *   **LASSO Regression:** Used for variable selection and regularization, trained with cross-validation.
    *   Evaluate each model using key classification metrics:
        *   Accuracy
        *   Precision
        *   Recall
        *   F1 Score
        *   AUC-ROC
    *   Visualize confusion matrices for each model.
    *   Visualize the Decision Tree and Random Forest feature importances.
    *   Generate ROC curves for all models for visual comparison.

5.  **Model Comparison:**
    *   Compile a performance comparison table for all models based on the evaluation metrics.
    *   Identify and justify the best-performing model based on the comparison.

## Models Implemented

*   Decision Tree
*   Logistic Regression
*   Neural Network (using Keras/TensorFlow)
*   Random Forest
*   LASSO Regression

## Evaluation Metrics

*   Accuracy
*   Precision
*   Recall
*   F1 Score
*   AUC-ROC

## Code Structure

The code is organized into distinct sections within a Jupyter Notebook or Google Colab environment:

*   Import necessary libraries.
*   Load the data.
*   Perform EDA.
*   Preprocess the data.
*   Build and evaluate each machine learning model.
*   Compare model performance.

## Getting Started

To run this notebook:

1.  Ensure you have the required libraries installed (`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `tensorflow`, `pyreadstat`). You can install them using `pip`:
2.  Make sure the dataset (`develop (1).sas7bdat`) is accessible, for example, by mounting your Google Drive in Google Colab.
3.  Run the code cells sequentially in a Jupyter Notebook or Google Colab.

## Results and Conclusion

Based on the evaluation metrics and visualizations, the Random Forest model demonstrated the best performance in predicting customer purchases for the variable annuity product. This model is recommended for use in the target marketing campaign.

## Author

NAVYA NANDURI

## License

MIT LICENSE
