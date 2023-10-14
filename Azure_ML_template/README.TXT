
This directory contains files (templates)
for a simple implementation of Machine Learning
using Azure ML Studio.

Files:

Two main scripts (train and predict):

    ml_train.py - script to train the model
        #   Get data from SQL database
        #   Clean data for training
        #   Train the model - save pkl file locally and to blob

    ml_predict.py - script to predict scores
        #   Get data from SQL database
        #   Get model pkl file from blob storage
        #   Clean data (same way as for training)
        #   Predict
        #   Save CSV file with predictions to the blob

There are three modules used in both scripts:

    mylib_bag.py - defines class MyBunch()

    mylib_sql.py - defines SQL queries

    mylib_ml_train.py - defines multiple functions and objects


The above two scripts can be run in ML Studio
and they also implemented as ML Pipelines.
(see/edit pipelines drafts in ML Studio Designer)

Note: 
    Our 3 modules should be zipped together 
    to be used in pipelines.
    This can be done from unix prompt.
    Go to the directory where modules are
    and run this command:

    zip -r ml_train_modules.zip mylib_bag.py mylib_sql.py mylib_ml_train.py
        
Note:
    We need specific version of the blob storage library.
    We run installation as part of both scripts:

    os.system(f"pip install azure-storage-blob==12.8.0")

