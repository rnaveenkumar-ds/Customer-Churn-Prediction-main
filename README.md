# Customer Churn Prediction and Retention Analysis using Machine Learning

## Project Overview

Customer churn is one of the biggest business challenges in the telecom industry. Retaining existing customers is significantly more cost-effective than acquiring new customers. This project focuses on analyzing telecom customer behavior and predicting customer churn using Machine Learning techniques.

The project combines:

- Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
- Feature Engineering
- Machine Learning Model Training
- Hyperparameter Tuning
- Model Evaluation
- Business Insights Generation

The final model uses the XGBoost Classifier to predict whether a customer is likely to stay, churn, or join.

---

# Problem Statement

Telecom companies lose revenue when customers discontinue services. The goal of this project is to:

- Analyze customer behavior patterns
- Identify factors influencing churn
- Build a predictive model to classify customer status
- Generate business insights to improve customer retention strategies

---

# Dataset Information

Dataset Used:
- Telecom Customer Churn Dataset

The dataset contains customer-related information such as:

- Customer demographics
- Subscription details
- Internet services
- Billing information
- Usage statistics
- Payment methods
- Contract types
- Customer status

---

# Technologies Used

## Programming Language
- Python

## Libraries & Frameworks

### Data Processing
- NumPy
- Pandas

### Data Visualization
- Matplotlib
- Seaborn
- Plotly

### Machine Learning
- Scikit-learn
- XGBoost

---

# Project Workflow

## 1. Importing Libraries

Essential Python libraries for:
- Data manipulation
- Visualization
- Machine learning
- Model evaluation

---

## 2. Loading Dataset

Dataset is loaded using Pandas:

```python
df = pd.read_csv('/content/telecom_customer_churn.csv')
```

---

## 3. Data Cleaning

### Removed Unnecessary Columns

The following columns were dropped because they do not significantly contribute to churn prediction:

- Customer ID
- Total Refunds
- Zip Code
- Latitude
- Longitude
- Churn Category
- Churn Reason

```python
df1.drop(['Customer ID','Total Refunds','Zip Code',
'Latitude','Longitude','Churn Category',
'Churn Reason'], axis='columns', inplace=True)
```

---

## 4. Handling Missing Values

Missing values were handled using interpolation and null value removal.

```python
df1 = df1.interpolate()
df1 = df1.dropna()
```

---

## 5. Exploratory Data Analysis (EDA)

Several visualizations were created to understand customer behavior.

### Visualizations Used

- Histogram plots
- Heatmaps
- Boxplots
- Density heatmaps
- Bar charts
- Correlation analysis

### Insights Generated

- Customer age distribution
- Churn behavior by age
- Contract impact on churn
- Payment method analysis
- Offer distribution
- Numerical feature relationships

---

# Data Visualization Examples

## Age Distribution Histogram

Used to analyze customer age groups.

```python
fig = px.histogram(df1, x='Age')
fig.show()
```

---

## Correlation Heatmap

Used to identify relationships between numerical features.

```python
sns.heatmap(data, annot=True)
```

---

## Customer Status Analysis

Compared:
- Stayed customers
- Churned customers
- Newly joined customers

---

# Feature Engineering

## Label Encoding

Converted categorical target labels into numerical values.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1.Customer_Status = le.fit_transform(df1.Customer_Status)
```

---

## Binary Encoding

Converted Yes/No categorical values into 1/0.

Example:

```python
df1.replace({'No':0,'Yes':1}, inplace=True)
```

---

## One-Hot Encoding

Applied to categorical features such as:

- Payment Method
- Contract
- Internet Type
- Offer
- City

```python
df1 = pd.get_dummies(data=df1,
columns=['Payment Method','Contract',
'Internet Type','Offer','City'])
```

---

# Feature Scaling

MinMaxScaler was used to normalize numerical columns.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])
```

---

# Train-Test Split

Dataset was split into:
- Training data (80%)
- Testing data (20%)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=5)
```

---

# Machine Learning Models Used

The following models were trained and evaluated:

| Model | Purpose |
|---|---|
| Random Forest | Ensemble learning |
| Logistic Regression | Linear classification |
| Gaussian Naive Bayes | Probabilistic classification |
| Decision Tree | Tree-based learning |
| XGBoost Classifier | Gradient boosting |

---

# Hyperparameter Tuning

GridSearchCV was used for finding the best parameters.

```python
from sklearn.model_selection import GridSearchCV
```

Cross-validation method:
- ShuffleSplit

```python
cv = ShuffleSplit(n_splits=5,
test_size=0.2,
random_state=0)
```

---

# Best Performing Model

## XGBoost Classifier

The XGBoost model achieved the highest performance among all tested models.

```python
reg = XGBClassifier()
reg.fit(X_train, y_train)
```

---

# Model Evaluation

## Accuracy Score

```python
accuracy_score(y_test, y_predicted)
```

---

## Confusion Matrix

Used to visualize:
- Correct predictions
- Incorrect predictions

```python
cm = confusion_matrix(y_test, y_predicted)
```

---

## Classification Report

Includes:
- Precision
- Recall
- F1-Score
- Support

```python
print(classification_report(y_test, y_predicted))
```

---

# Business Insights

Key insights identified from the analysis:

- Customers with shorter contracts are more likely to churn.
- Customers using specific payment methods show higher churn rates.
- Higher monthly charges increase churn probability.
- Long-term customers are more likely to stay.
- Customer offers influence retention behavior.
- Internet service type impacts churn trends.

---

# Challenges Faced

- Handling categorical variables
- Managing high-dimensional data after encoding
- Feature scaling
- Model selection
- Hyperparameter tuning
- Data preprocessing

---

# Future Improvements

The project can be further improved by:

- Using SMOTE for class imbalance handling
- Implementing Deep Learning models
- Performing feature selection
- Building a deployment pipeline
- Creating a real-time churn prediction dashboard
- Deploying using Flask or Streamlit
- Using ROC-AUC analysis
- Adding feature importance analysis

---

# Conclusion

This project demonstrates the complete Machine Learning workflow for customer churn prediction in the telecom industry.

The analysis helps businesses:
- Understand customer behavior
- Predict churn risks
- Improve retention strategies
- Increase customer satisfaction
- Reduce revenue loss

The XGBoost model provided strong predictive performance and valuable business insights.

---

# Learning Outcomes

Through this project, the following concepts were learned and implemented:

- Data preprocessing
- Feature engineering
- Exploratory Data Analysis
- Machine Learning model training
- Hyperparameter tuning
- Model evaluation
- Business insight generation

---

# Author

Naveen Kumar

Data Science & AI/ML Enthusiast

---

# License

This project is open-source and available for educational purposes.
