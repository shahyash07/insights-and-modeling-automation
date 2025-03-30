from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import storage
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import kagglehub
from kagglehub import KaggleDatasetAdapter
import re
from airflow.utils.task_group import TaskGroup

# DAG
dag = DAG(
    'train_and_store_models_parallel',
    default_args={
        'start_date': datetime(2024, 1, 1),
        'retries': 3,
        'retry_delay': timedelta(minutes=15)
    },
    schedule_interval=None,
    catchup=False,
)

# GCS path 
GCS_PATH = "gs://us-central1-airflow-71403677-bucket/data"

def data_extraction_transform():
    """Output: DataFrame,
     Extracts and transforms sales data from Kaggle dataset with feature engineering."""
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    file_path_list = [f"Sales_data/Sales_{i}_2019.csv" for i in months ]
    df = pd.DataFrame()
    for i in file_path_list:
        # print(i)
        # Load the latest version
        df_i = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "pigment/big-sales-data",
            i,
        )
        df = pd.concat([df, df_i],ignore_index=True)
    
    df=df.dropna()
    df = df[df['Quantity Ordered']!= 'Quantity Ordered'] 
    df['Month'] = pd.to_datetime(df['Order Date'], format='%m/%d/%y %H:%M').dt.month_name()
    df['dayofweek'] = pd.to_datetime(df['Order Date'], format='%m/%d/%y %H:%M').dt.dayofweek
    df['week'] = pd.to_datetime(df['Order Date'], format='%m/%d/%y %H:%M').dt.isocalendar().week
    # plan is to see sales of products across addresses
    df['postal code'] = df['Purchase Address'].str[-8:].str.replace(' ', '_')
    # df['City'] = df['Purchase Address'].apply(lambda x: x.split(',')[1].strip())
    df['sales'] = df['Quantity Ordered'].astype(float) * df['Price Each'].astype(float)
    df = df[['Product', 'sales', 'dayofweek', 'week' , 'postal code', 'Month']]
    df = pd.get_dummies(df, columns=['postal code', 'Month'],dtype=int)

    return df

def model_building_and_validation(df, product):
    """Input: DataFrame, product name,
    Trains and validates RandomForest model for a specific product."""
    df_model_data = df[df['Product'] == product].drop(columns=['Product'])
    X = df_model_data.drop(columns=['sales'])
    y = df_model_data['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # model_dict[product] = model # storing the models for further usage in the notebook
    joblib.dump(model, f"{re.sub('[^a-zA-Z0-9]', '_', product)}_RF_model.pkl")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Training model for {product} with training data {len(X_train)}")
    print(f"diffrent sales amount in the data {y_test.unique()}")
    print(f"Test data length {len(X_test)}")
    print(f"Mean Squared Error for {product}: {mse}")
    print(f"R-squared for {product}: {r2}")



# Function to extract and transform data
def extract_and_transform_data(**kwargs):
    """Input: Airflow context kwargs, 
    Output: CSV file path,
    Extracts data, saves to GCS, and pushes file path to XCom."""
    df = data_extraction_transform()
    kwargs['ti'].xcom_push(key='gcs_sales_file_path', value=f'{GCS_PATH}/sales_data_all_products.csv')

    return df.to_csv(f'{GCS_PATH}/sales_data_all_products.csv')  # saving it in gcs

def upload_to_gcs(model_file, gcs_path):
    """Input: model file name, GCS path
     Uploads model file to Google Cloud Storage.
     """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('us-central1-airflow-71403677-bucket')  
    blob = bucket.blob(f"{gcs_path}/{model_file}")
    blob.upload_from_filename(model_file)
    print(f"Model {model_file} uploaded to GCS at {gcs_path}/{model_file}")

# Function to get unique products from the data
def get_unique_products(**kwargs):
    """
      Output: List of products,
     Gets unique products from data and pushes to XCom.
     """
    ti = kwargs['ti']
    df_file_path = ti.xcom_pull(task_ids='extract_and_transform_data', key='gcs_sales_file_path')
    df = pd.read_csv(df_file_path)
    products = df['Product'].unique().tolist()
    # Push the products list to XCom
    ti.xcom_push(key='products_list', value=products)
    return products

# Task to extract and transform data
extract_task = PythonOperator(
    task_id='extract_and_transform_data',
    python_callable=extract_and_transform_data,
    provide_context=True,
    dag=dag,
)


# Task to get unique products
get_products_task = PythonOperator(
    task_id='get_unique_products',
    python_callable=get_unique_products,
    provide_context=True,
    dag=dag,
)

# Function to build model for a single product
def build_model_for_product(**kwargs):
    """Input: Airflow context kwargs with product name,
      Builds and validates model for a specific product using data from XCom."""
    ti = kwargs['ti']
    product = kwargs['product']
    df_file_path = ti.xcom_pull(task_ids='extract_and_transform_data', key='gcs_sales_file_path')
    df = pd.read_csv(df_file_path)
    model_building_and_validation(df, product)
    upload_to_gcs(f"{re.sub('[^a-zA-Z0-9]', '_', product)}_RF_model.pkl", GCS_PATH)


with dag:
    with TaskGroup(group_id='model_training_tasks') as model_training_tasks:
        for i in range(20):
            task_id = f'build_model_{i}'
            build_task = PythonOperator(
                task_id=task_id,
                python_callable=build_model_for_product,
                op_kwargs={'product': f'{{{{ task_instance.xcom_pull(task_ids="get_unique_products", key="products_list")[{i}] }}}}'},
                trigger_rule='all_success',
                pool='model_training_pool', # this make sure to run it parrellaly with task groups
            )

# Setting up a pool for parallel task execution
dag.pool_size = 10  # smaller airflow image so just getting 10 so that composer doesnt crashes

# Create start and end dummy operators
start = DummyOperator(
    task_id='start',
    dag=dag
)

end = DummyOperator(
    task_id='end',
    dag=dag
)

# Set up task dependencies
start >> extract_task >> get_products_task >> model_training_tasks >> end
