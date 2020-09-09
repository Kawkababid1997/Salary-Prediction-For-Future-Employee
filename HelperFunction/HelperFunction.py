#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# load the data
def load_dataset(file):
    """
    Read a comma-separated values (csv) file into DataFrame.
    
    Parameters
    ----------
    file: the directory of the dataset
    """
    return pd.read_csv(file)

#Merge Dataset
def consolidate_data(df1, df2, key=None, left_index=False, right_index=False):
    """
    Performs inner join to return only records that are present in both dataframe
    
    Parameters
    ----------
    
    df1: DataFrame1
    df2: DataFrame2
    key: Column or index level names to join on. These must be found in both DataFrames.
    
    left_index: bool, default False
                Use the index from the left DataFrame as the join key(s). If it is a
                MultiIndex, the number of keys in the other DataFrame (either the index
                or a number of columns) must match the number of levels.
    
    right_index: bool, default False
                 Use the index from the right DataFrame as the join key. Same caveats as left_index.
    
    """
    return pd.merge(left=df1, right=df2, how='inner', on=key, left_index=left_index, right_index=right_index)
    

# Check for duplicates
def check_for_duplicates(*dataframes):
    """
    Return the number of duplicates in a dataframe
    
    Parameters
    ----------
    dataframes: Name of the dataframes
    
    """
    print("Hey Helper Function Check Duplicates")
    for dataframe in dataframes:
        #name =[x for x in globals() if globals()[x] is dataframe][0]
        print(f' Duplicates in {dataframe.__name__} = {dataframe.duplicated().sum()}')
        

# Visualize a single column of a DataFrame
def visualize_column(data, bins=10):
    """
    Returns a boxplot and displot of a single columns from a dataset
    
    Parameters
    ----------
    data : DataFrame, array, or list of arrays Dataset for plotting. 
    
    bins : argument for matplotlib hist(), or Default is 10, optional
           Specification of hist bins, or None to use Freedman-Diaconis rule.
    
    """
    plt.figure(figsize = (14,6))
    plt.subplot(1,2,1)
    sns.boxplot(data)
    plt.subplot(1,2,2)
    sns.distplot(data, bins)
    plt.show()

# Statistical Reports of a given data
def statistical_reports(data):
    """
    Generate descriptive statistics that summarize the central tendency,
    dispersion and shape of a dataset's distribution, excluding ``NaN`` values
    
    Analyzes both numeric and object series, as well as ``DataFrame`` 
    column sets of mixed data types. The output will vary depending on what is provided
    
    Also provides the potential outliers (lower bounds and upper bounds).
    
    Parameters
    ----------
    data : DataFrame, array, or list of arrays Dataset
    
    """
    stat = data.describe()
    print(stat)
    IQR = stat['75%'] - stat['25%']
    upper = stat['75%'] + 1.5*IQR
    lower = stat['25%'] - 1.5*IQR
    print('The upper and lower bounds for suspected outliers are {} and {}.'.format(upper, lower))

# Clean Data
def clean_data(raw_df, duplicate_column_check, target_column, rearrange = False):
    """
    Remove rows that contain value of target_column (salary in this case) < 0 
    and check duplicate based on a single column
    
    Parameters
    ----------
    
    raw_df: Dataset that are targeted to be cleaned
    duplicate_column_check: name the column on the raw_df that may have duplicates
    target_column: predicted column
    rearrange: Default False. 
               Shuffle and reindex data -- shuffling improves cross-validation
    """
    clean_df = raw_df.drop_duplicates(subset= duplicate_column_check)
    clean_df = clean_df[clean_df[target_column]>0]
    if rearrange == False:
        return clean_df
    else:
        return shuffle(clean_df).reset_index()
    
# Plot Feature
def plot_feature(df, col):
    """
    Make plot for each features
    left, the distribution of samples on the feature
    right, the dependance of salary on the feature
    
    Parameters
    ----------
    df : DataFrame, array, or list of arrays Dataset
    col: columns of a dataframe
    
    """
    plt.figure(figsize = (14, 6))
    plt.subplot(1, 2, 1)
    if df[col].dtype == 'int64':
        df[col].value_counts().sort_index().plot()
    else:
        #change the categorical variable to category type and order their level by the mean salary
        #in each category
        mean = df.groupby(col)['salary'].mean()
        df[col] = df[col].astype('category')
        levels = mean.sort_values().index.tolist()
        df[col].cat.reorder_categories(levels, inplace=True)
        df[col].value_counts().plot()
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel('Counts')
    
    plt.subplot(1, 2, 2)
    if df[col].dtype == 'int64' or col == 'companyId':
        #plot the mean salary for each category and fill between the (mean - std, mean + std)
        mean = df.groupby(col)['salary'].mean()
        std = df.groupby(col)['salary'].std()
        mean.plot()
        plt.fill_between(range(len(std.index)), mean.values-std.values, mean.values + std.values, \
                         alpha = 0.1)
    else:
        sns.boxplot(x = col, y = 'salary', data=df)
    
    plt.xticks(rotation=45)
    plt.ylabel('Salaries')

# encode label
def encode_label(df, col):
    '''
    encode the categories using average salary for each category to replace label
    
    Parameters
    ----------
    df: Name of the dataframes
    col: column in the dataframe
    '''
    
    cat_dict ={}
    cats = df[col].cat.categories.tolist()
    for cat in cats:
        cat_dict[cat] = df[df[col]==cat]['salary'].mean()
    df[col] = df[col].map(cat_dict)
    
def label_Encoder(df1, df2 = None, cat_vars=None, num_vars=None, engingeering=False):
    """
    Perfoms one-hot encoding on all categrorical variables and combines result with continuous variable
    Parameters
    ----------
    df1, df2: Select the Dataframe
    cat_vars: categorical variables
    num_vars: numerical variables
    engingeering: Default False
    """
    if engingeering == False:
        cat_df = pd.get_dummies(df1[cat_vars])
        num_df = df1[num_vars].apply(pd.to_numeric)
        return pd.concat([cat_df, num_df], axis=1)#ignore_index = False
    else:
        le = LabelEncoder()
        le.fit(df1[cat_vars])
        df1[cat_vars] = le.transform(df1[cat_vars])
        df2[cat_vars] = le.transform(df2[cat_vars])
        return df1[cat_vars], df2[cat_vars], le   
    
# train a model
def train_model(model, feature_df, target_df):
    """
    Train the sklearn-estimatator
    Parameters
    ----------
    model: estimator object implementing 'fit'
                The object to use to fit the data.

    feature_df : array-like of shape (n_samples, n_features)
                 The data to fit. Can be for example a list, or an array.

    target_df : The target variable to try to predict in the case of supervised learning.
    """
    neg_mse = cross_val_score(model, feature_df, target_df, cv = 2, n_jobs = 2, scoring = 'neg_mean_squared_error')
    mean_mse = (-1)* np.mean(neg_mse)
    cv_std = np.std(neg_mse)
    print('\nModel:\n', model)
    print('Average MSE:\n', mean_mse)
    print('Standard deviation during cross validation:\n', cv_std)

# Get the target variable
def get_target_df(df, target_variable):
    """
    returns the target dataframe
    Parameters
    ----------
    df: dataframe
    target_variable: name the target variable
    
    """
    return df[target_variable]

# Feature Engineering Function
def FeatureEng_GrpAvg(df,cat_cols,target_col):
    '''Returns descriptive statistics by aggregating target variable '''
    groups = df.groupby(cat_cols)
    group_stats_df = pd.DataFrame({'group_mean': groups[target_col].mean()})
    group_stats_df['group_max'] = groups[target_col].max()
    group_stats_df['group_min'] = groups[target_col].min()
    group_stats_df['group_std'] = groups[target_col].std()
    group_stats_df['group_median'] = groups[target_col].median()
    group_stats_cols = group_stats_df.columns.tolist()
    return group_stats_df, group_stats_cols

def FeatureEng_merge(df, new_df, cat_cols, group_stats_cols, fillna = False):
    ''' Merges the dataframe of the new engineered features with train/test dataframe '''
    new_ds = pd.merge(df, new_df, on = cat_cols, how = 'left')
    if fillna:
        for col in group_stats_cols:
            new_ds[col] = [0 if math.isnan(x) else x for x in new_ds[col]]
    return new_ds

def shuffle_df(df):
    ''' Shuffles the dataframe '''
    return shuffle(df).reset_index()

def cross_val_model(model, feature_df, target_col, n_procs, mean_mse, cv_std):
    ''' Cross validates the model for 50% (cv=2) of the training set'''
    neg_mse = cross_val_score(model, feature_df, target_col, cv = 2, n_jobs = n_procs, scoring = 'neg_mean_squared_error')
    mean_mse[model] = -1.0 * np.mean(neg_mse)
    cv_std[model] = np.std(neg_mse)

# Print summary
def print_summary(model, mean_mse, cv_std):
    ''' Prints a short summary of the model performance '''
    print('\nmodel:\n', model)
    print('Average MSE:\n', mean_mse[model])
    print('Standard deviation during cross validation:\n', cv_std[model])

# Feature Importances
def get_model_feature_importances(model, feature_df):
    ''' Gets and sorts the importance of every feature as a predictor of the target '''
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = [0] * len(feature_df.columns)
    
    feature_importances = pd.DataFrame({'feature': feature_df.columns, 'importance': importances})
    feature_importances.sort_values(by = 'importance', ascending = False, inplace = True)
    ''' set the index to 'feature' '''
    feature_importances.set_index('feature', inplace = True, drop = True)
    return feature_importances

# save results
def save_results(model, mean_mse, predictions, feature_importances):
    with open('model.txt', 'w') as file:
        file.write(str(model))
    feature_importances.to_csv('feature_importances')
    np.savetxt('predictions.csv', predictions, delimiter=',')
