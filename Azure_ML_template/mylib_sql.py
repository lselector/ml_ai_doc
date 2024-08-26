
"""
# module containing a dictionary for running different SQL queries
# keys - labels (query_training or query_predict)
# values - SQL strings
# Usage:
# from mylib_sql import get_dict_sql
#
# dict_sql = get_dict_sql()
# sql = dict_sql["query1"]
#   updated October 8, 2021
"""

def get_dict_sql():
    dict_sql = {}

    # ----------------------------------------------------------
    dict_sql["query_training"] = """

SELECT 
    row_id,
    mydate
    some_value

FROM 
    some_table
WHERE
    ....
"""

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    dict_sql["query_predict"] = """

SELECT 
    row_id,
    mydate
    some_value

FROM 
    some_table
WHERE
    ....
"""

    # ----------------------------------------------------------

    return dict_sql

