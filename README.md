# Portuguese-Bank-Marketing
A machine learning project that analyses Portuguese bank marketing campaigns and predicts customer subscriptions to term deposits using Python, data visualization, and multiple ML models, including XGBoost.




# Portuguese Bank Marketing — Term Deposit Prediction

## Project Overview

This project analyses historical marketing campaigns conducted by a Portuguese banking institution to promote term deposits. It uses exploratory data analysis and machine learning to identify customer and campaign patterns and predict the likelihood of a customer subscribing to a term deposit.

The project was developed as part of the PRCP-1000 Portuguese Bank Marketing capstone project.

## Objectives

- Perform data cleaning and exploratory data analysis.
- Identify patterns associated with term-deposit subscriptions.
- Train and compare multiple machine learning models.
- Develop a prediction interface for exploring customer subscription probabilities.
- Provide data-driven recommendations to improve marketing campaign planning.

## Dataset

The project uses the `bank-additional-full.csv` dataset, which contains information about customer demographics, contact details, previous campaign outcomes, economic indicators, and the target variable `y`.

- Original campaign period: May 2008 to November 2010.
- Target: Whether the customer subscribed to a term deposit (`yes` or `no`).
- Duplicate records were removed during preprocessing.
- The `duration` feature was excluded from the main predictive model because it is known only after a call and would cause data leakage for pre-call predictions.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook
- ipywidgets

## Machine Learning Models

The following models were evaluated:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost
5. Tuned XGBoost

### Best Candidate Model: Tuned XGBoost

On the current test set, tuned XGBoost achieved:

| Metric | Result |
|---|---:|
| Accuracy | 90.00% |
| Precision | 64.86% |
| Recall | 24.46% |
| F1-score | 35.52% |
| ROC-AUC | 81.53% |

Although tuned XGBoost achieved the highest accuracy and ROC-AUC among the five evaluated models, its recall was relatively low at the default prediction threshold. Random Forest achieved higher recall and F1-score. The final model and decision threshold should therefore depend on the bank's marketing objectives and the cost of missed potential subscribers.

## Project Features

- Data cleaning and preprocessing
- Exploratory data analysis and visualizations
- Comparison of five machine learning models
- Feature-importance analysis
- Interactive customer prediction interface
- Sample customer explorer
- Marketing analytics dashboard
- Business insights and recommendations
- Challenges, limitations, and conclusion

## Repository Structure

```text
Portuguese-Bank-Marketing/
├── README.md
├── portuguese_bank_marketing.ipynb
├── xgboost_bank_marketing.pkl
├── app.py
└── bank-additional-full.csv
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Saiyaswin/Portuguese-Bank-Marketing.git
cd Portuguese-Bank-Marketing
```

### 2. Install the required packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost ipywidgets jupyter
```

If you use the separate application in `app.py`, check its imports and install any additional dependencies it requires.

### 3. Run the Jupyter Notebook

```bash
jupyter notebook portuguese_bank_marketing.ipynb
```

Run the notebook cells in order. Ensure the dataset and saved model are available at the paths expected by the code.

## Key Business Recommendations

- Prioritize customer segments with promising historical subscription patterns.
- Consider previous campaign outcomes when planning future outreach.
- Compare contact methods and campaign periods using historical subscription rates.
- Avoid excessive repeated contacts without evaluating their effectiveness.
- Adjust prediction thresholds according to marketing costs and the need to identify more potential subscribers.
- Validate the model on later-period data and controlled marketing campaigns before production deployment.

## Limitations

The model was evaluated using a stratified random train-test split of historical data. Performance may differ on future campaigns because customer behaviour and economic conditions can change. Predictions indicate estimated likelihood, not guaranteed customer behaviour.

## Author

**Saiyaswin Vechham**

GitHub: [Saiyaswin](https://github.com/Saiyaswin)

## Disclaimer

This project is intended for educational and analytical purposes. Model predictions should support, rather than replace, business judgment and further validation.
