"""
Import libraries
"""
import sys
import traceback
import pandas as pd
from sqlalchemy import create_engine
from bokeh.plotting import figure, output_file, show
#from bokeh.io import output_notebook
import math
import numpy as np

print("Libraries are loaded...")


def circle_area(r):
    print(type(r))
    if type(r) not in [int, float]:
        raise TypeError("The radius must be a non-negative real number!")

    if r < 0:
        raise ValueError("The radius cannot be negative!")

    return math.pi * (r ** 2)

def create_database_dataframe():
    """
    Task
    1)Read the needed CSV-files (train.csv, test.csv, ideal.csv) with a pandas framework
    2)Create a database and crate and copy the read in csv-files into tables
    -->("ideal_functions","train_functions","test_functions")
    3)Create a working data framework for each table in the database
    -->("ideal_functions","train_functions","test_functions")

    Input: train.csv, test.csv, ideal.csv in the current directionary
    Output: df_test, df_train, df_ideal
    """

    # Engine of local sqlite databas in the current directionary
    engine = create_engine('sqlite:///Functions.db')
    # Read in the csv data and create a database with sqlalchemy (if the table already exists it will be rplaces with the new one)
    # ideal data
    df_ideal_csvtosql = pd.read_csv("ideal.csv", delimiter=",")
    table_name = "ideal_functions"

    df_ideal_csvtosql.to_sql(
        table_name,
        engine,
        if_exists='replace',
        index=False,
        chunksize=1  # loading line by line
    )

    # read from table ideal functions
    df_ideal = pd.read_sql_table(
        table_name,
        con=engine)
    # df_ideal.shape

    # train data
    df_train_csvtosql = pd.read_csv("train.csv", delimiter=",")
    table_name = "train_functions"
    df_train_csvtosql.to_sql(
        table_name,
        engine,
        if_exists='replace',
        index=False,
        chunksize=1  # loading line by line
    )

    # read from table train_functions to pandas dataframe
    df_train = pd.read_sql_table(
        table_name,
        con=engine)
    # df_train.shape

    # test data
    df_test_csvtosql = pd.read_csv("test.csv", delimiter=",")
    table_name = "test_functions"
    df_test_csvtosql.to_sql(
        table_name,
        engine,
        if_exists='replace',
        index=False,
        chunksize=1  # loading line by line
    )

    # read from table train_functions to pandas dataframe
    df_test = pd.read_sql_table(
        table_name,
        con=engine)
    # df_test.shape

    #Exception handling
    if not isinstance(df_test, pd.DataFrame):
        raise TypeError("df_test must be a pandas dataframe")
    if not isinstance(df_train, pd.DataFrame):
        raise TypeError("df_train must be a pandas dataframe")
    if not isinstance(df_ideal, pd.DataFrame):
        raise TypeError("df_test must be a pandas dataframe")
    if df_test.empty:
        raise ValueError("The dataframe df_test is empty!!!")
    if df_train.empty:
        raise ValueError("The dataframe df_train is empty!!!")
    if df_ideal.empty:
        raise ValueError("The dataframe df_ideal is empty!!!")


    print("Data is read from csv-files and stored in the database Functions...")
    print("Dataframes for further processing are created...")
    return df_test, df_train, df_ideal


def get_ideal_functions(df_train, df_ideal):
    """
    This functions identifies the ideal function fitting the most to the train functions regarding the criterion sum of squared y-deviations

    Input: df_train, df_ideal
    Output:

    """
    #Exception handling for input values
    if type(df_train) != pd.DataFrame:
        raise TypeError("The needed input parameter df_train is not a pandas dataframe")
    if df_train.empty:
        raise ValueError("The dataframe df_train is empty!!!")
    if type(df_ideal) != pd.DataFrame:
        raise TypeError("The needed input parameter df_ideal is not a pandas dataframe")
    if df_ideal.empty:
        raise ValueError("The dataframe df_ideal is empty!!!")


    SSE_dict = {}
    max_dev_dict = {}
    mapping_criterion_dict = {}
    list_idealfct_dict = {}
    # Bokeh Output in jupyter notebook
    #output_notebook() # remove # if charts are supposed not to be displayed in the browse

    print("The ideal functions for each train function is:")
    for i_train in range(1, 5):
        name_y_train = "y" + str(i_train)
        for i_ideal in range(1, 51):
            name_y_ideal = "y" + str(i_ideal)
            SSE = 0
            max_dev = 0
            for x_counter in range(0, 400):
                if x_counter == 0:
                    x = -20
                else:
                    x = -20 + 40 / x_counter
                SSE += (df_train[name_y_train][x_counter] - df_ideal[name_y_ideal][x_counter]) * (
                            df_train[name_y_train][x_counter] - df_ideal[name_y_ideal][x_counter])
                if max_dev < (df_train[name_y_train][x_counter] - df_ideal[name_y_ideal][x_counter]):
                    max_dev = (df_train[name_y_train][x_counter] - df_ideal[name_y_ideal][x_counter])
            max_dev_dict[i_ideal] = max_dev
            SSE_dict[i_ideal] = SSE

        nr_of_ideal_function = min(SSE_dict, key=lambda k: SSE_dict[k])
        MinSSE = SSE_dict[nr_of_ideal_function]
        mapping_criterion_dict[i_train] = math.sqrt(2) * max_dev_dict[nr_of_ideal_function]
        name_y = "y" + str(nr_of_ideal_function)
        p = figure(title="Mapping the ideal function ({}) to the train function ({})".format(name_y,name_y_train), toolbar_location="above",
           plot_width=600, plot_height=300)
        #in jupyter notebook use legend instead of legend_label
        p.line(df_train["x"], df_train[name_y_train], color='darkblue', line_width=2, legend_label="Train function: " + name_y_train)
        p.line(df_ideal["x"], df_ideal[name_y], color='peru', line_width=2, legend_label="Ideal function: " + name_y)
        p.xaxis.axis_label = 'X'
        p.yaxis.axis_label = 'Y'
        show(p)
        list_idealfct_dict[i_train] = nr_of_ideal_function
        print("The chosen ideal function for the train function: " + str(name_y_train) + " is: " + str(
            nr_of_ideal_function) + " with the minimized sum of all y-deviations squared: " + str(MinSSE))

    #Exception handling
    if type(list_idealfct_dict) !=dict:
        raise TypeError("The output list_idealfct_dict is not a dictionary!")
    if not bool(list_idealfct_dict):
        raise ValueError("The list with the ideal functions assigned to the train functions is empty!")

    if type(mapping_criterion_dict) !=dict:
        raise TypeError("The output mapping_criterion_dict is not a dictionary!")
    if not bool(mapping_criterion_dict):
        raise ValueError("The list with the mapping criterion is empty!")

    return list_idealfct_dict, mapping_criterion_dict


def apply_idealfct_to_testfct(df_test, df_ideal, list_idealfct_dict, mapping_criterion_dict):
    """
    Program use the test data provided (B) to determine for each and every x-y-pair of
    values whether or not they can be assigned to the four chosen ideal functions**; if so, the program also
    needs to execute the mapping and save it together with the deviation at hand

    Input: df_test,df_ideal, list_idealfct_dict, mapping_criterion_dict
    Output: FinalTest.csv, table testdata_mapped_with_idealfct in database Functions.db

    """

    #Exception handling of input data
    if type(list_idealfct_dict) !=dict:
        raise TypeError("The output list_idealfct_dict is not a dictionary!")
    if not bool(list_idealfct_dict):
        raise ValueError("The list with the ideal functions assigned to the train functions is empty!")
    if type(mapping_criterion_dict) !=dict:
        raise TypeError("The output mapping_criterion_dict is not a dictionary!")
    if not bool(mapping_criterion_dict):
        raise ValueError("The list with the mapping criterion is empty!")
    if type(df_test) != pd.DataFrame:
        raise TypeError("The needed input parameter df_test is not a pandas dataframe")
    if df_test.empty:
        raise ValueError("The dataframe df_test is empty!!!")
    if type(df_ideal) != pd.DataFrame:
        raise TypeError("The needed input parameter df_ideal is not a pandas dataframe")
    if df_ideal.empty:
        raise ValueError("The dataframe df_ideal is empty!!!")


    count_test_row = df_test.shape[0]  # gives number of row count

    # Creating dataframe for final resualts
    df_final_test_data = pd.DataFrame(np.nan, index=np.linspace(0, 99, 100),
                                      columns=["X (test func)", "Y (test func)", "Delta Y (test func)",
                                               "No_of_ideal_func"])
    df_final_test_data["X (test func)"] = df_test["x"]
    df_final_test_data["Y (test func)"] = df_test["y"]
    df_final_test_data["No_of_ideal_func"] = df_final_test_data["No_of_ideal_func"].astype(str)

    for i_test in range(0, count_test_row):
        for i_ideal in range(1, 5):
            name_idealfunction = "y" + str(list_idealfct_dict[i_ideal])
            #print("The mapping criterion for the ideal function {} is: {}".format(name_idealfunction, mapping_criterion_dict[i_ideal]))
            x_value = df_test["x"][i_test]

            for x_idealcount in range(0, 400):
                if x_value == df_ideal["x"][x_idealcount]:
                    Dev_Test_Ideal = abs(df_test["y"][i_test] - df_ideal[name_idealfunction][x_idealcount])
                    if Dev_Test_Ideal < mapping_criterion_dict[i_ideal]:
                        df_final_test_data.loc[i_test, "No_of_ideal_func"] = name_idealfunction
                        df_final_test_data.loc[i_test, "Delta Y (test func)"] = Dev_Test_Ideal

    print("The chosen ideal functions were maped to the test function recording the mapping criterion...")

    # Draw the mapped chosen ideal funcion and the test dataset in a chart
    p = figure(title="Mapping the chosen ideal function to the test dataset", toolbar_location="above",
           plot_width=1200, plot_height=600)
    p.circle(df_final_test_data["X (test func)"], df_final_test_data["Y (test func)"], color='deepskyblue', line_width=1, legend_label="Test data points")
    for i_ideal in range(1,5):
        name_idealfunction = "y" + str(list_idealfct_dict[i_ideal])
        x_values=df_final_test_data.query('No_of_ideal_func == @name_idealfunction')["X (test func)"].tolist()
        y_values=df_final_test_data.query('No_of_ideal_func == @name_idealfunction')["Y (test func)"].tolist()
        if i_ideal==1:
            c="lime"
        if i_ideal==2:
            c="mediumvioletred"
        if i_ideal==3:
            c="chocolate"
        if i_ideal==4:
            c="slategray"
        p.circle(x=x_values, y=y_values, color=c, line_width=3, legend_label="Train function: " + name_idealfunction)
    p.xaxis.axis_label = 'X'
    p.yaxis.axis_label = 'Y'
    show(p)

    table_name = "testdata_mapped_with_idealfct"
    engine = create_engine('sqlite:///Functions.db')
    df_final_test_data.to_sql(
        table_name,
        engine,
        if_exists='replace',
        index=False,
        chunksize=1  # loading line by line
    )
    df_final_test_data.to_csv("FinalTest.csv", index=False, encoding='utf-8')

    print("Mapped ideal functions to the test function is stored in the database Functions.db in the table {}...".format(table_name))

def get_exception_info():
    """
    Task of the function get_exception_info is to print informations about occuring exceptions
    Input: n.a.
    Output: exception infos as a string
        - Name of the file
        - Line Number
        - Procedure Name
        - Line Code

    """
    try:
        exception_type, exception_value, exception_traceback = sys.exc_info()
        file_name, line_number, procedure_name, line_code = traceback.extract_tb(exception_traceback)[-1]
        exception_info = "File Name: {}\nLine Number: {}\nProcedure Name: {}\nLine Code:{}".format(file_name,
                                                                                                   line_number,
                                                                                                   procedure_name,
                                                                                                   line_code)
    except:
        pass
    return exception_info


def main():
    """
    The main function is running the programm in order to fulfill the given task
    """

    # Read in the needed csv files, create a database with the corresponding tables
    # and create working dataframes for further processing
    try:
        df_test, df_train, df_ideal = create_database_dataframe()
    except:
        exception_info = get_exception_info()
        print(exception_info)
    finally:
        pass

    # Identify the ideal functions for the four ideal functions

    try:
        list_idealfct_dict, mapping_criterion_dict = get_ideal_functions(df_train, df_ideal)
    except:
        exception_info = get_exception_info()
        print(exception_info)
    finally:
        pass

    # Mapp the identified ideal functions to the test function and create a table with the mapped values in the database Functions.db
    try:
        apply_idealfct_to_testfct(df_test, df_ideal, list_idealfct_dict, mapping_criterion_dict)
    except:
        exception_info = get_exception_info()
        print(exception_info)
    finally:
        pass
    print("Finished task :)")

if __name__ == '__main__':
    main()