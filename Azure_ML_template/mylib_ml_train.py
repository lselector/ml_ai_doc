
"""
# mylib_ml_train.py
# functions for model
# October 8, 2021
"""

import os, sys, re, time
import pandas as pd
import numpy as np
import datetime as dt

# --------------------------------------------------------------
def date_time():
    """
    # returns string YYYY-MM-DD HH:MM:SS
    """
    # import datetime as dt
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return now_str

# --------------------------------------------------------------
def print_date_time(label = "date_time"):
    """
    # prints current date and time
    """
    now_str = date_time()
    print (label, " : ", now_str)

# --------------------------------------------------------------
def sec_to_hms(secs):
    """
    # converts seconds into string in format "HH:MM:SS"
    """
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    h_str = "%d"   % int(h)
    m_str = "%02d" % int(m)
    s_str = "%02.2f" % round(s,2)
    s_str = re.sub(r'\.00$','',s_str)
    return ':'.join([h_str,m_str,s_str])

# --------------------------------------------------------------
def download_file_from_blob(bag, local_dir=None, myfile=None):
    print("in download_file_from_blob()")
    os.makedirs(local_dir, exist_ok=True)         # make sure directory exists
    local_path = os.path.join(local_dir, myfile)  # full file name
    container_name = "machinelearning"     # hardcoded ? XXXXXXXXXXX

    print("create blob_client for specific file names")
    print(f"  container_name = {container_name}")
    print(f"  fname_remote   = {myfile}")
    print(f"  local_dir      = {local_dir}")
    print(f"  fname_local    = {myfile}")
    blob_client = bag.blob_service_client.get_blob_client(
        container=container_name, blob=myfile)

    print(f"download from container {container_name} file {myfile}")
    print(f"to local file {local_path}")
    with open(local_path, "wb") as fh:
        mystream = blob_client.download_blob()
        fh.write(mystream.readall())

# --------------------------------------------------------------
def upload_file_to_blob(bag, local_dir=None, myfile=None):
    print("in upload_file_to_blob()")
    os.makedirs(local_dir, exist_ok=True) # make sure directory exists
    local_path = os.path.join(local_dir, myfile)
    container_name = "machinelearning"    # hardcoded ? XXXXXXXXXXX
 
    print("create blob_client for specific file names")
    print(f"  container_name = {container_name}")
    print(f"  fname_remote   = {myfile}")
    print(f"  local_dir      = {local_dir}")
    print(f"  fname_local    = {myfile}")
    blob_client = bag.blob_service_client.get_blob_client(
        container=container_name, blob=myfile)
    
    print(f"upload local file {myfile} to blob")
    print(f"container {container_name} file {myfile}")
    with open(local_path, "rb") as fh:
        blob_client.upload_blob(fh, overwrite=True)


# ---------------------------------------------------------------
def print_separator():
    for ii in range(5):
        print("-"*80)

# ---------------------------------------------------------------
def clean_1(df):
    pass
    return df

# ---------------------------------------------------------------
def clean_2(df):
    pass
    return df

# ---------------------------------------------------------------
def clean_3(df):
    pass
    return df

# ---------------------------------------------------------------
def clean_dataset(df):
    """
    # Procedure to clean the raw dataset:
    # - filter obsrvations
    # - impute missing values
    # - generate features
    """
    df = clean_1(df)
    df = clean_2(df)
    df = clean_3(df)

    return df

# ---------------------------------------------------------------
def select_columns(df, columns):
    """ Select columns of interest """
    return df.loc[:, columns]

# ---------------------------------------------------------------
NUMERIC_FEATURES = ['f1','f2','f3']

CATEGORICAL_FEATURES = ['c1','c2']

target = ['t1']

# ---------------------------------------------------------------
