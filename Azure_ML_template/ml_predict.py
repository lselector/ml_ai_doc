"""
# predict scores
#   Get data from SQL database
#   Get model pkl file from blob storage
#   Clean data (same way as for training)
#   Predict
#   Save CSV file with predictions to the blob
#   updated 10/08/2021
"""

# import standard ML modules
import os, sys, joblib
os.system(f"pip install azure-storage-blob==12.8.0")
import pandas as pd
import numpy as np
import datetime as dt

from sklearn.metrics import f1_score, classification_report, \
     roc_auc_score, make_scorer, roc_curve, \
     accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
# --------------------------------------------------------------
import mylib_bag
from mylib_bag import *

import mylib_ml_train
from mylib_ml_train import *

import mylib_sql
from mylib_sql import get_dict_sql
# --------------------------------------------------------------

from azureml.core import Workspace, Dataset, Datastore
from azureml.data.datapath import DataPath
from azureml.core.authentication import InteractiveLoginAuthentication
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

# --------------------------------------------------------------
def part0_authenticate(bag):
    print("in part0_authenticate()")    
    # interactive_auth = InteractiveLoginAuthentication(
    #     tenant_id = "11111-2222-33333-444444-555555") # XXXXXXXXXXXXXXX
    # -------------------------------------------
    from azureml.core.workspace import Workspace
    from azureml.core.authentication import ServicePrincipalAuthentication

    sp = ServicePrincipalAuthentication(
        tenant_id="11111-2222-33333-444444-555555",   # XXXXXXXXXXXXXXX
        service_principal_id="11111-2222-33333-444444-555555",   # XXXXXXXXXXXXXXX
        service_principal_password="11111-2222-33333-444444-555555"   # XXXXXXXXXXXXXXX
    )

    # load existing worksapce
    bag.ws = Workspace.get(name='machine_learning_2020',
                   auth=sp,
                   subscription_id='11111-2222-33333-444444-555555',   # XXXXXXXXXXXXXXX
                   resource_group='BusinessIntelligence'
    )

# --------------------------------------------------------------
def part1_blob_service_client(bag):
    print("in part1_blob_service_client()")
    bag.ss_key="123123123123123123123123123123123123132123123123123123123123123123123123"
    mylist=[]
    mylist.append("DefaultEndpointsProtocol=https")
    mylist.append("AccountName=MyAccountName")
    mylist.append("AccountKey=MyVeryLongAccountKey")
    mylist.append("EndpointSuffix=core.windows.net")
    bag.ss_connection_string=";".join(mylist)
    bag.blob_service_client = BlobServiceClient.from_connection_string(bag.ss_connection_string)

# --------------------------------------------------------------
def part2_get_data(bag) :
    print("in part2_get_data()")    

    print("create datastore object for our SQL DW")

    datastore = Datastore.get(bag.ws, 'edw_sqldw_prod')
 
    print("get sql from query from python module")    
    dict_sql = get_dict_sql()
    sql = dict_sql["query_predict"]
    # sql = """select top 10 * from sometable"""  # XXXXXXXXXXXXX
    print(sql)   # it is a long multi-line SQL query string

    print("start running SQL query - usually takes ~ ... sec")
    t_before = time.time()
    query = DataPath(datastore, sql)
    bag.tabular_data = Dataset.Tabular.from_sql_query(query, query_timeout=600)
    t_after = time.time()
    t_seconds = t_after - t_before
    print(f"getting data took {t_seconds:.2f} seconds")
    print("-"*40, "after running query")

# --------------------------------------------------------------
def part3_convert_to_df(bag):
    print("in part3_convert_to_df()")
    print("convert tabular_data object into Pandas DataFrame - takes ... sec")
    t_before = time.time()
    bag.df = bag.tabular_data.to_pandas_dataframe()
    t_after = time.time()
    t_seconds = t_after - t_before
    print(f"converting to Pandas DataFrame took {t_seconds:.2f} seconds")
    
    print(f"number of rows: {len(bag.df):,d}") # approx ... rows
    print(f"number of columns: {len(bag.df.columns):d}")
    print("-"*40,"columns:")
    for col in bag.df.columns:
        print(col)
    print(bag.df.head(3))
    print("-"*60)
    # ------------------------------------
    local_dir="data"
    os.makedirs('data', exist_ok=True)
    myfile="DailyInput.csv"
    data_path = f"{local_dir}/{myfile}"
    print(f"Save data locally to {data_path}")
    bag.df.to_csv(data_path, index=False)
    # ------------------------------------
    print('Upload {data_path} to BlobStorage')
    upload_file_to_blob(bag, local_dir=local_dir, myfile=myfile)
    print("DONE upload")

# --------------------------------------------------------------
def part4_prep_features(bag):
    print("in part4_prep_features")
    print('clean bag.df')
    bag.df1 = clean_dataset(bag.df.copy())
    print('select columns of interest')
    bag.df1 = select_columns(bag.df1, 
            ['LeadID'] + CATEGORICAL_FEATURES+NUMERIC_FEATURES+target)
    # ------------------------------------
    # get model from blob storage
    download_file_from_blob(bag, 
        local_dir="models", myfile="model_bag.pkl")
    model_bag_path = "models/model_bag.pkl"
    print(f"get model from pkl file {model_bag_path}")
    with open(model_bag_path, 'rb') as fh:
        bag.bag2 = joblib.load(fh)

# --------------------------------------------------------------
def part5_predict(bag):
    print("in part3_predict")
    col_list = bag.bag2.col_list  # col_list of the model
    df_dummy = pd.get_dummies(bag.df1, columns=CATEGORICAL_FEATURES)
    df_dummy = df_dummy.reindex(columns=col_list).fillna(0.0) 
    df_inference = pd.concat([select_columns(bag.df1, NUMERIC_FEATURES), df_dummy], axis = 1)
    df_inference.shape
    print('Predict .... ')
    y_pred_proba = bag.bag2.model.predict_proba(df_inference)
    scored_proba = y_pred_proba[:, 1].round(7)
    lead_id      = bag.df1['MyDate'].values        # XXXXXXXXXXXXXXX  which column?
    print("Creating output DataFrame df_result")
    bag.df_result = pd.DataFrame(
        data = { "MyDate" : lead_id,
                 "Scored Probabilities" : scored_proba},
        columns = ["MyDate", "Scored Probabilities"])
    print(bag.df_result.head())
    print("-"*60)

# --------------------------------------------------------------
def part6_save_and_upload_results(bag):
    print("in part4_save_and_upload_results")
    os.makedirs('data', exist_ok=True)
    result_path = "data/DailyScores.csv"
    print(f"Save the results locally to {result_path}")
    bag.df_result.to_csv(result_path, index=False)
    print('Upload Inference Results to BlobStorage')
    upload_file_to_blob(bag, 
        local_dir="data", 
        myfile="DailyScores.csv")
    print("DONE")

# --------------------------------------------------------------
def azureml_main(dataframe1 = None, dataframe2 = None):
    """
    # run predictions
    """
    print_separator()
    t_start = time.time()
    print_date_time("START")

    bag = MyBunch()

    part0_authenticate(bag)
    part1_blob_service_client(bag)
    part2_get_data(bag)
    part3_convert_to_df(bag)
    part4_prep_features(bag)
    part5_predict(bag)
    part6_save_and_upload_results(bag)

    t_end = time.time()
    mydiff = round(t_end - t_start,2)
    print_date_time("FINISHED")
    print(f"Elapsed time = {sec_to_hms(mydiff)}")
    print_separator()
    return pd.DataFrame()

# --------------------------------------------------------------
if __name__ == "__main__":
    azureml_main()
