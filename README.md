# insights-and-modeling-automation

# Electronics Sales Data analytics
- 2019 sales data from USA is analysed for electronics items
- insights found can be seen from the presentation and electronics_sales_data_insights.ipynb file
- models built for top 5 products including ETL, feature engineering, model building
- here this files is built and scaled to automate using airflow deployed in gcp
- The pipeline is built to scale and uses multiple pool resources from airflow
-  pending: for the dag: I would create a separate module with all the functions and then call all the functions as a best practice
- here, artifacts are stored in gcs bucket
- pending: yml file configuration for github actions
-   This process would copy the dag from github to gcs dag folder and then after pushing the changes it can trigger the Dag
-   I have extensively used my Pc financial deployed dags:  to take reference

# Diabetics analytics
- Insights are found using Kaggle data from 2015 for people who don't have diabetes, pre-diabetics, and diabetic patients
- across healthcare factors
- Statistical measures, Random forest implementation for finding the risk factors
- Statistical tests conducted to see the BMI relation to Diabetics
- Details can be found in diabetis_data_analysis.ipynb file
