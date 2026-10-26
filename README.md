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
## Week 2

### Building and Evaluating Machine Learning Models

In Week 2, I built and evaluated several machine learning models to predict customer churn using the Telco Customer Churn dataset.

The dataset contained 7,043 customers and 21 original columns. After preprocessing and one-hot encoding, the feature table contained 30 columns. The data was divided into 5,634 training records and 1,409 testing records. The churn rate was 26.54% in both sets.

I first created a majority-class baseline. It achieved 73.46% accuracy but had 0% churn recall because it predicted that every customer would stay.

The main models produced these results:

| Model | Accuracy | Churn Precision | Churn Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.807 | 0.658 | 0.567 | 0.609 | 0.842 |
| Balanced Logistic Regression | 0.740 | 0.507 | 0.786 | 0.616 | 0.841 |
| Decision Tree | 0.796 | 0.634 | 0.545 | 0.586 | 0.827 |
| Random Forest | 0.806 | 0.672 | 0.527 | 0.591 | 0.843 |

The random forest achieved the highest ROC-AUC at 0.843, while logistic regression was easier to interpret. The balanced logistic regression detected more churners, increasing recall from 0.567 to 0.786, but its precision decreased because it produced more false alarms.



## Week 3: Model Optimization and Unsupervised Learning

In Week 3, I studied how much model performance can change because of random train-validation splits. Using 20 different random seeds, Logistic Regression accuracy ranged from 0.780 to 0.828, with a standard deviation of 0.0104.

I then used five-fold stratified cross-validation to obtain more reliable estimates.

| Model | Cross-validated AUC |
|---|---:|
| Logistic Regression | 0.846 +/- 0.013 |
| Random Forest | 0.844 +/- 0.011 |
| XGBoost | 0.8502 +/- 0.0117 |

I compared validation curves, grid search, and random search. Grid search achieved a best AUC of 0.8468 in 72 seconds, while random search achieved 0.8464 in 76 seconds.

The final model was selected using cross-validation. Tuned XGBoost was selected because it had the highest CV AUC. Its test AUC, evaluated once at the end, was 0.8483.

K-means produced four customer segments. The riskiest segment was the New, high-cost customer group, which had a churn rate of 43%. This segment should receive early loyalty offers and onboarding support.

PCA showed that 15 of the 30 components explain 90% of the variance. This indicates that some encoded features contain overlapping information.

The main lesson from Week 3 was that a single split can be misleading, so model comparisons should use cross-validation and the test set should remain untouched until the final evaluation.

I also tested a business-based probability threshold. Since missing a churner costs PKR 6,000 and contacting a non-churner costs PKR 1,000, the calculated threshold was 0.1429. At this threshold, churn recall increased to 0.920 and the estimated cost decreased from PKR 1,082,000 to PKR 619,000 on the test data.

The main lesson from Week 2 was that accuracy alone is not enough for churn prediction. Recall, precision, ROC-AUC, and business cost are also important when deciding which customers should receive retention offers.
## Week 3: Model Optimization and Unsupervised Learning

In Week 3, I studied how much model performance can change because of random train-validation splits. Using 20 different random seeds, Logistic Regression accuracy ranged from 0.780 to 0.828, with a standard deviation of 0.0104.

I then used five-fold stratified cross-validation to obtain more reliable estimates.

| Model | Cross-validated AUC |
|---|---:|
| Logistic Regression | 0.846 +/- 0.013 |
| Random Forest | 0.844 +/- 0.011 |
| XGBoost | 0.8502 +/- 0.0117 |

I compared validation curves, grid search, and random search. Grid search achieved a best AUC of 0.8468 in 72 seconds, while random search achieved 0.8464 in 76 seconds.

The final model was selected using cross-validation. Tuned XGBoost was selected because it had the highest CV AUC. Its test AUC, evaluated once at the end, was 0.8483.

K-means produced four customer segments. The riskiest segment was the New, high-cost customer group, which had a churn rate of 43%. This segment should receive early loyalty offers and onboarding support.

PCA showed that 15 of the 30 components explain 90% of the variance. This indicates that some encoded features contain overlapping information.

The main lesson from Week 3 was that a single split can be misleading, so model comparisons should use cross-validation and the test set should remain untouched until the final evaluation.
## Week 4: From Notebook to Product

I packaged the Week 3 XGBoost churn model into a Streamlit web application called Churn Risk Advisor.

The application allows a user to:

- Enter one customer's information
- View the predicted churn probability
- View a LOW, WATCH, or HIGH risk band
- Change the contact threshold
- Test what-if changes such as contract type
- Upload a CSV file for batch scoring
- Download the scored customer file

### Model Card

**Intended use:** Rank existing telecom customers by churn risk so a retention team can prioritize follow-up calls.

**Not for:** Credit decisions, pricing decisions, or denying services.

**Dataset:** IBM Telco Customer Churn, 7,043 customers, approximately 26.5% churn.

**Model:** Tuned XGBoost, selected using five-fold cross-validation.

**Performance:** CV AUC 0.8502 +/- 0.0117; test AUC 0.8483. The test set was used once.

**Threshold:** 0.1429, based on the Week 2 cost analysis using a false-negative cost of PKR 6,000 and a false-positive cost of PKR 1,000.

**Limitations:** The model was trained on one US telecom dataset from one period. It has not been validated on Pakistani telecom data. Its relationships are associations from the dataset and should not be interpreted as proof of cause.

**Owner:** Muhammad Haissam  
**Version:** 1.0  
**Date:** October 2026  


# Week 5 — Neural Networks and Image Classification

**Student:** Muhammad Haissam  
**Course:** Applied AI  
**University:** Quaid-i-Azam University, Islamabad  
**Department:** Electronics  
**Semester:** 2

## Overview

For this lab, I trained a model to classify landscape images into six categories. I used a pretrained ResNet18 model, first training its new classification layer and then fine-tuning its final block.

## Dataset

The project uses the [Intel Image Classification dataset on Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

The dataset contains six classes:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

| Class | Training images | Test images |
|---|---:|---:|
| Buildings | 2,191 | 437 |
| Forest | 2,271 | 474 |
| Glacier | 2,404 | 553 |
| Mountain | 2,512 | 525 |
| Sea | 2,274 | 510 |
| Street | 2,382 | 501 |
| **Total** | **14,034** | **3,000** |

I split the training images into 11,929 training images and 2,105 validation images. The separate test set contains 3,000 images.

## Approach

I used ResNet18 with pretrained ImageNet weights. Images were resized to 224 × 224 pixels and normalized. During training, I also used random crops, horizontal flips, and small brightness and contrast changes.

I trained the model in two stages:

1. **Feature extraction:** I froze the pretrained layers and trained the new classification layer for three epochs. The best validation accuracy was **90.6%**.
2. **Fine-tuning:** I unfroze ResNet18’s final block and trained it with the classification layer. The notebook contains two sequential three-epoch fine-tuning runs. The best validation accuracy from the later run was **93.9%**.

## Test results

The final model achieved **90.1% test accuracy** on the 3,000 test images.

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Buildings | 90.6% | 90.4% | 90.5% |
| Forest | 98.1% | 99.2% | 98.6% |
| Glacier | 88.5% | 77.6% | 82.7% |
| Mountain | 80.8% | 86.7% | 83.6% |
| Sea | 92.1% | 96.7% | 94.4% |
| Street | 91.6% | 91.8% | 91.7% |

Glacier was the hardest class for the model. The most frequent confusion was a glacier image classified as a mountain image, which happened 94 times. Some glacier and mountain scenes share similar visual features, which may have contributed to these errors.

## Files

- `week5-neural-networks.ipynb` — the notebook containing the data inspection, model training, learning curves, test evaluation, and error analysis.

## How to run

Open the notebook in Kaggle, attach the Intel Image Classification dataset, and select a GPU accelerator. Run the notebook cells in order.

## Reflection

Fine-tuning improved the model’s validation accuracy compared with training only the classification layer. The test results were strong overall, but the lower glacier recall shows that the model still has difficulty distinguishing some glacier scenes from mountains.
