
"""
# Train model
#   Get data from SQL database
#   Clean data for training
#   Train the model - save pkl file locally and to blob
#   updated 10/08/2021
"""

# import standard ML modules
import os, sys, time, joblib
os.system(f"pip install azure-storage-blob==12.8.0")
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
    
    datastore = Datastore.get(bag.ws, 'MyDataStore')
 
    print("get sql from query from python module")    
    dict_sql = get_dict_sql()
    sql = dict_sql["query_training"]
    print("\n\n")
    print(sql)   # it is a long multi-line SQL query string
    print("\n\n")
    print("start running SQL query - usually takes ~ 16 sec")
    t1 = time.time()
    query = DataPath(datastore, sql)
    bag.tabular_data = Dataset.Tabular.from_sql_query(query, query_timeout=600)
    seconds = time.time() - t1
    print(f"query took {seconds:.1f} sec")

# --------------------------------------------------------------
def part3_convert_to_df(bag):
    print("in part3_convert_to_df()")
    t1 = time.time()
    print("convert tabular_data object into Pandas DataFrame - takes 6 sec")
    bag.df = bag.tabular_data.to_pandas_dataframe()
    seconds = time.time() - t1
    print(f"conversion took {seconds:.1f} sec")
    
    print(f"number of rows: {len(bag.df):,d}") # approx 55K rows
    print(f"number of columns: {len(bag.df.columns):d}")
    print("-"*40,"columns:")
    for col in bag.df.columns:
        print(col)
    print("-"*40,"head(3):")
    print(bag.df.head(3))
    print("-"*60)

# --------------------------------------------------------------
def part4_prep_features(bag):
    print("in part4_prep_features()")
    print('clean df')
    bag.df1 = clean_dataset(bag.df.copy())
    
    print('select columns of interest')
    bag.df1 = select_columns(bag.df1, CATEGORICAL_FEATURES + NUMERIC_FEATURES + target)

    print('create unique categorical column:value dictionary')
    unique_values = {column: bag.df1[column].dropna().unique() for column in CATEGORICAL_FEATURES}
    col_list = [f'{key}_{value}' for key in unique_values.keys() for value in unique_values[key]]

    print('create dummy variables from column:value dictionary')
    df_dummy = pd.get_dummies(bag.df1, columns=CATEGORICAL_FEATURES)
    df_dummy = df_dummy.reindex(columns=col_list).fillna(0.0) 
    bag.df_final = pd.concat([select_columns(bag.df1, NUMERIC_FEATURES), df_dummy], axis = 1)
    bag.col_list = col_list
    print(f"df_final number of rows: {len(bag.df_final):,d}")               # ... rows
    print(f"df_final number of columns: {len(bag.df_final.columns):d}")     # ... columns

# --------------------------------------------------------------
def part5_train_model(bag):
    print("in part5_train_model()")

    print('Split data into train and test set for metrics evaluation')
    X_train, X_test, y_train, y_test = train_test_split(
        bag.df_final, 
        select_columns(bag.df1, target),
        test_size=0.1,
        random_state=42,
        stratify=select_columns(bag.df1, target)
    )
    print(f"X_train.shape = {X_train.shape}")
    print(f"X_test.shape  = {X_test.shape}")
    print(f"y_train.shape = {y_train.shape}")
    print(f"y_test.shape  = {y_test.shape}")
    print("-"*40)
    t_before_fit = time.time()
    print('Fit Random Forest Classifier') # takes ... seconds
    model_rf = RandomForestClassifier(
        class_weight={0:2, 1:20},
        # class_weight="balanced", 
        max_depth=8, 
        min_samples_leaf=4, 
        min_samples_split=10,
        n_estimators=200)

    model_rf.fit(X_train, y_train.values.ravel())
    t_after_fit = time.time()
    fit_seconds = t_after_fit - t_before_fit
    print(f"fitting model took {fit_seconds:.2f} seconds")
    print("---------- after fit ----------")
    print( model_rf.get_params() )
    print("-"*20)
    expected  = y_train
    predicted = model_rf.predict(X_train)
    print(classification_report(expected, predicted))
    print("-"*20)
    confusion = confusion_matrix(expected, predicted)
    print(confusion)
 
    print("-"*40,'Estimate the model on test dataset')
    y_test_pred = model_rf.predict(X_test)
    y_test_pred_proba = model_rf.predict_proba(X_test)

    fpr, tpr, threshold = roc_curve(y_test, y_test_pred_proba[:, 1])

    roc_auc = roc_auc_score(y_test, y_test_pred_proba[:, 1])
    print("-"*40, f"{roc_auc:.4f}")  # 0.94

    print("-"*40, "confusion matrix")
    confusion = confusion_matrix(y_test, y_test_pred)
    print(confusion)

    print("-"*40, "feature importance")

    feature_importance = pd.DataFrame(
        data=model_rf.feature_importances_,
        columns=['importance'], 
        index=X_train.columns
    )

    feature_importance.sort_values(by='importance', 
        ascending=False, inplace=True)

    print(feature_importance[:30].to_string())

    print("-"*40, "now re-train on the whole data")

    X_train, X_test, y_train, y_test = train_test_split(
        bag.df_final, 
        select_columns(bag.df1, target),
        test_size=0.01,
        random_state=42,
        stratify=select_columns(bag.df1, target)
    )
    print(f"X_train.shape = {X_train.shape}")
    print(f"X_test.shape  = {X_test.shape}")
    print(f"y_train.shape = {y_train.shape}")
    print(f"y_test.shape  = {y_test.shape}")
    print("-"*40)
    print('Fit Random Forest Classifier') # takes about 20 seconds
    model_rf = RandomForestClassifier(
        class_weight={0:2, 1:20},
        # class_weight="balanced", 
        max_depth=8, 
        min_samples_leaf=4, 
        min_samples_split=10,
        n_estimators=200)

    model_rf.fit(X_train, y_train.values.ravel())
    print("-"*40,"after fit")
    print( model_rf.get_params() )
    print("-"*20)
    expected  = y_train
    predicted = model_rf.predict(X_train)
    print(classification_report(expected, predicted))
    print("-"*20)
    confusion = confusion_matrix(expected, predicted)
    print(confusion)

    print("-"*40,"returning the model")

    bag.model_rf = model_rf

# --------------------------------------------------------------
def part6_save_model(bag):
    print("in part6_save_model()")
    bag2 = MyBunch()
    bag2.model = bag.model_rf
    bag2.col_list = bag.col_list
    
    os.makedirs('models', exist_ok=True)
    bag_path = 'models/model_bag.pkl'
    print(f"Saving bag2 to {bag_path} ...")
    with open(bag_path, 'wb') as fd:
        joblib.dump(value=bag2, filename=bag_path)
    print("DONE")

# --------------------------------------------------------------
def part7_upload_model_to_blob(bag):
    print("in part7_upload_model_to_blob()")
    upload_file_to_blob(bag, 
        local_dir="models", 
        myfile="model_bag.pkl")
    print("DONE")

# --------------------------------------------------------------
def azureml_main(dataframe1 = None, dataframe2 = None):
    """
    # run training
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
    part5_train_model(bag)
    part6_save_model(bag)
    part7_upload_model_to_blob(bag)

    t_end = time.time()
    mydiff = round(t_end - t_start,2)
    print_date_time("FINISHED")
    print(f"Elapsed time = {sec_to_hms(mydiff)}")
    print_separator()
    return pd.DataFrame()


if __name__ == "__main__":
    azureml_main()
