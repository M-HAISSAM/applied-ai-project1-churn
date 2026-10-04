# Applied AI Project 1: Customer Churn

**Student:** Muhammad Haissam  
**University:** Quaid-i-Azam University, Islamabad  
**Department:** Electronics

This repository contains my work for Weeks 1–4 of Applied AI Project 1. The project starts by exploring customer churn data, then moves into building, improving, and deploying prediction models.

## Week 1

### Exploratory Data Analysis

For Week 1, I explored customer details, subscribed services, and billing information to identify patterns associated with churn.

**Notebook:** [week1-eda.ipynb](week1-eda.ipynb)

### Dataset

- Source: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Customer records: 7,043
- Columns: 21, including customer ID and churn status
- Target column: `Churn`, where Yes means the customer left and No means they stayed

### Data Quality

`TotalCharges` was stored as text. Converting it to numbers revealed 11 missing values, all belonging to customers with zero months of tenure.

I retained these records in the main dataset and excluded them from the total-charges plots. Correlations were calculated using available values for each pair of fields.

There were no duplicate rows or duplicate customer IDs.

### Analysis Completed

The notebook includes a numerical summary, missing-value checks, churn counts, and seven visualization groups:

1. Customer tenure
2. Monthly charges
3. Total charges
4. Contract type
5. Internet service
6. Payment method
7. Correlation heatmap

### Main Findings

- **26.54% of customers churned:** 1,869 left, while 5,174 stayed.
- Customers who churned had a median tenure of **10 months**, compared with **38 months** for customers who stayed.
- Median monthly charges were **$79.65** for churners and **$64.43** for stayers.
- Month-to-month contracts had a churn rate of **42.71%**, compared with **11.27%** for one-year contracts and **2.83%** for two-year contracts.
- Fiber optic customers had a churn rate of **41.89%**, compared with **18.96%** for DSL customers.
- Electronic check users had the highest churn rate among payment methods at **45.29%**.

These results describe associations in this dataset. They do not establish why individual customers left.

### Tools Used

Python, Pandas, NumPy, Matplotlib, and Seaborn. The notebook was developed and run on Kaggle.

### Running the Notebook

1. Download `week1-eda.ipynb` and import it into Kaggle.
2. Attach the Telco Customer Churn dataset published by BlastChar.
3. Check that `dataset_path` matches the attached CSV's location.
4. Run the notebook from top to bottom.

### Next Steps

In Week 2, I will test how well these customer characteristics predict churn and compare machine learning models. I will also examine how model evaluation changes when the two churn classes have unequal sizes.
