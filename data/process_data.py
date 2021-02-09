'''
DESCRIPTION

This module imports, processes and cleans raw data.
The output is a transformed data to used for a ML Pipeline

INPUTS

messages_filepath -   path containing the messages csv data file
categories_filepath - path containing the categories csv data file

OUTPUTS

Saves the combined processed data as a SQLite database

SCRIPT EXECUTION SAMPLE

python process_data.py disaster_messages.csv  disaster_categories.csv  Disaster_Response_Data.db

'''

#import relevant libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    DESCRIPTION
    Loads and combines raw data files at given file path
    
    INPUTS
    messages_filepath -   path containing the messages csv data file
    categories_filepath - path containing the categories csv data file
    
    OUTPUTS
    df - Dataframe with combined data
    
    '''
    # read in messages csv file as dataframe
    messages = pd.read_csv(messages_filepath)
    
    # read in categories csv file as dataframe
    categories = pd.read_csv(categories_filepath)
    
    # merge the messages and categories into one dataframe
    df = pd.merge(messages, categories, on='id')
    
    
    return df


def clean_data(df):
    
    '''
    DESCRIPTION
    Cleans and transforms data
    
    INPUTS
    df - loaded and combined dataframe
    
    OUTPUTS
    df -cleaned and transformed dataframe
    
    '''
    # Splits each row of the category dataframe into separate columns
    categories = df['categories'].str.split(";", expand=True)
    
    # get first row of categories df
    row = categories.iloc[:1]
    
    # extract unique column names from selected row
    category_colnames = row.apply(lambda x: x.str[:-2]).values.tolist()[0]
    # rename the columns of the categories df
    categories.columns = category_colnames
    
    # convert all category values to just numerical 0's and 1's
    for column in categories:
        categories[column] = categories[column].astype(str).apply(lambda x: x[-1:])
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from df
    df = df.drop(['categories'], axis=1)
    # concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)
    
    # find and remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    DESCRIPTION
    Saves processed dataframe to a SQLite Database
    
    INPUTS
    df - cleaned and processed dataframe
    database_filename - preferred name for Database
    
    '''
    engine = create_engine('sqlite:///' + database_filename )
    
    # exports dataframe to SQL table
    df.to_sql('messages', engine, index=False, if_exists='replace')
      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()