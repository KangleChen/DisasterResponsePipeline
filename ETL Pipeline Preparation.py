#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline Preparation

# import libraries
import pandas as pd
from sqlalchemy import create_engine

# load messages dataset
messages = pd. read_csv('messages.csv')
messages.head()

# load categories dataset
categories = pd. read_csv('categories.csv')
categories.head()

# merge datasets
df = messages.merge(categories, how= 'inner', on = 'id')
df.head()

# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(pat = ';', expand = True)
categories.head()

# select the first row of the categories dataframe
row = categories.iloc[0,:]

# use this row to extract a list of new column names for categories.
category_colnames = [row_name.split('-')[0] for row_name in row]

# rename the columns of `categories`
categories.columns = category_colnames
categories.head()


# ### 4. Convert category values to just numbers 0 or 1.

for column in categories:
    # set each value to be the last character of the string
    categories[column] = [item[-1] for item in categories[column]]
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])
categories.head()


# ### 5. Replace `categories` column in `df` with new category columns.

# drop the original categories column from `df`
df.drop('categories', axis = 1 )
df.head()

# concatenate the original dataframe with the new `categories` dataframe
df=pd.concat([df, categories], axis = 1)
df.head()


# ### 6. Remove duplicates.

# drop duplicates
df = df.drop_duplicates()

# ### 7. Save the clean dataset into an sqlite database.

engine = create_engine('sqlite:///disaster.db')
df.to_sql('disaster', engine, index=False)




