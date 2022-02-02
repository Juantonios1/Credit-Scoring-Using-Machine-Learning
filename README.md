# Loan-Repayment-Analysis
To know which borrower will repay or not so the company can prevent the loss because of the borrower not repay and keep the borrower who can repay still can lend the money.

![Churn](churn.jpeg)

For full process, please visit my portofolio on <a href="https://github.com/Juantonios1/Churn-Analysis-with-Telco-Customer-Churn-Dataset/blob/main/Churn%20Analysis%20with%20Telco%20Customer%20Churn%20Dataset%20Final.ipynb">Churn Analysis</a>.  

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Content</summary>
  <ol>
    <li>
      <a href="#business-background">Business Background</a>
    </li>
    <li>
      <a href="#data-understanding">Data Understanding</a>
    </li>
    <li>
      <a href="#exploratory-data-analysis">Exploratory Data Analysis</a>
    </li>
    <li><a href="#data-analytics">Data Analytics</a></li>
    <li><a href="#data-preprocessing">Data Preprocessing</a></li>
    <li><a href="#model-selection">Model Selection</a></li>
    <li><a href="#explainable-and-interpretable-machine-learning">Explainable and Interpretable Machine Learning</a></li>
    <li><a href="#conclusion-and-recommendation">Conclusion and Recommendation</a></li>
    <li><a href="#contributors">Contributors</a></li>
  </ol>
</details>

## Business Background
**Context :**  
The company is one of the big provider of telecommunication services and have a lot of customer. The company have services include: phone service, TV streaming, internet and internet security and make profit from customer who paid their services monthly. Churn is a measurement of the percentage of accounts that cancel or choose not to renew their subscriptions. A high churn rate can negatively impact Monthly Recurring Revenue (MRR) and can also indicate dissatisfaction with a product or service. The source of this dataset is from <a href="https://www.kaggle.com/blastchar/telco-customer-churn">Kaggle</a>.  

**Problem Statement :**  
The company wants to know how much loss is caused by churn and want to prevent customer churn but we don't know which customer will leaving and we dont know why. If we know which customer will churn, we can give a voucher to keep customer or we will allocate the money to improve the services who make customer churns. the company wants to know how much loss is caused by churn.

**Goals :**  
To find how much loss caused by churn and predict which customer will churn and find out the reason so company can do action which give the company profit.

**Metric Evaluation :**    
Determine the suitable metric to rate the performance from the model

## Data Understanding

| Feature      	| Description                                                                                                                                                                                                               	|
|--------------	|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| customerID         	| Unique ID for customer                                                                                                                                                                                                           	|
| gender      	| Male or Female                                                                                                                                                                                 	|
| SeniorCitizen     	| Senior citizen or not (>55)                                                                                                                                                                               	|
| Partner          	| Has a partner or not                                                                                                                                                                                                         	|
| Dependents        	| Has dependents or not                     	|
| tenure       	| Number of months the customer has stayed with the company / business                                                                                                                                                                                         	|
| PhoneService   	|Has a phone service or not	|
| MultipleLines        	| Has multiple lines or not                                                                                                                                                                                                          	|
| InternetService         	| Kind of internet                                                                                                                                                      	|
|OnlineSecurity    	| Whether the customer has online security or not	|
|OnlineBackup         	| Has online backup or not                                                                                                                                                                                                            	|
|DeviceProtection         	| Has device protection or not                                                                                                                                                     	|
| TechSupport     	|Has tech support or not                                                                                                                                                     	|
|StreamingTV         	| Has streaming TV or not                                                                                                                                                  	|
| StreamingMovies     	|Has streaming movies or not    
| Contract       	| The contract term of the customer                                                                                                                                                                                                         	|
| PaperlessBilling 	| Gender of candidate      
| Contract       	| The contract term of the customer                                                                                                                                                                                                         	|
| PaperlessBilling 	| Has paperless billing or not
|PaymentMethod| The customer’s payment method 
| MonthlyCharges      	| The amount charged to the customer monthly                                                                                                                                                                                                         	|
| TotalCharges  	| The total amount charged to the customer
| Churn 	|  The measure of customers who stop using a product

## Exploratory Data Analysis
At this stage, a brief analysis of the data will be carried out, as follows:
* Distribution Data Numerical
* Data Cardinalities
* Data Correlation
* Missing Values
* Data Imbalance
* Identify Outliers
* Identify Duplicates

## Data Analytics
At this stage, another information analysis will be carried out, as follows:
* Loss caused by Churn
* Data Proportion based on Target
* Independent Test with Chi-squared
* Senior Citizen Feature

## Data Preprocessing
At this stage, data preparation and processing will be carried out before being used as a data model, as follows:
* Missing Value
* Transformer
* Splitting Data

## Model Selection
At this stage will be done making and optimizing the machine learning model, as follows:
* Model Benchmark
* Feature Selection
* Imbalance Method
* Hyperparameter Tuning

## Explainable and Interpretable Machine Learning
At this stage there will be an explanation in the decision making of the machine learning model, in the following ways:
* SHAP 

## Conclusion 
We conclude our result and give recommendation based on it
* Summary Model
* Business Insight
* Conclusion

## Contributors:
Juan Antonio Suwardi - antonio.juan.suwardi@gmail.com  
