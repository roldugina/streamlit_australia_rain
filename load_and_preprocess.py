import joblib

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Tuple

def load_model_components(model_path):
    loaded_model = joblib.load(model_path)
    return {
        'num_imputer' : loaded_model['num_imputer'],
        'cat_imputer': loaded_model['cat_imputer'],
        'scaler' : loaded_model['scaler'],
        'encoder' : loaded_model['encoder'],
        'model' : loaded_model['model'],
        'input_cols' : loaded_model['input_cols'],
        'target_col' : loaded_model['target_col'],
        'numeric_cols' : loaded_model['numeric_cols'],
        'categorical_cols' : loaded_model['categorical_cols'],
        'encoded_cols' : loaded_model['encoded_cols']}

def preprocess_data(raw_df: pd.DataFrame, delete_cols: List[str]) -> Tuple[Dict[str, Any], MinMaxScaler, OneHotEncoder, List[str]]:
    """
    Preprocess the raw DataFrame for machine learning.
    
    Parameters:
    raw_df (pd.DataFrame): The raw input DataFrame.
    scaler_numeric (bool): Whether to apply MinMaxScaler to numeric columns.
    delete_cols (List[str]): List of columns to delete from the input features.
    
    Returns:
    Tuple[Dict[str, Any], MinMaxScaler, OneHotEncoder, List[str]]: A dictionary containing preprocessed training and validation data,
    the scaler used, the encoder used, and the list of input columns.
    """
         
    # Define input, target, numeric, categorical columns
    input_cols, target_col, numeric_cols, categorical_cols = define_columns(raw_df, delete_cols)
    
    # Split the data into training and validation sets
    train_df, val_df, test_df = split_data(raw_df, target_col)
   
    # Separate inputs and targets
    train_inputs, train_targets = create_inputs_targets(train_df, input_cols, target_col)
    val_inputs, val_targets = create_inputs_targets(val_df, input_cols, target_col)
    test_inputs, test_targets = create_inputs_targets(test_df, input_cols, target_col)

    X_train_xgb = train_inputs[numeric_cols + categorical_cols]
    X_val_xgb = val_inputs[numeric_cols + categorical_cols]
    X_test_xgb = test_inputs[numeric_cols + categorical_cols]
        
    # Scale numeric columns
    train_inputs, val_inputs, test_inputs, num_imputer, scaler = preprocess_numeric(train_inputs, val_inputs, test_inputs, numeric_cols)
                
    # Encode categorical columns
    train_inputs, val_inputs, test_inputs, cat_imputer, encoder, encoded_cols = preprocess_categorical(train_inputs, val_inputs, test_inputs, categorical_cols)
    
    # Prepare final training and validation sets
    X_train = train_inputs[numeric_cols + encoded_cols]
    X_val = val_inputs[numeric_cols + encoded_cols]
    X_test = test_inputs[numeric_cols + encoded_cols]
    
    
    
    return  {
        'X_train': X_train,
        'y_train': train_targets,
        'X_val': X_val,
        'y_val': val_targets,
        'X_test': X_test,
        'y_test': test_targets,
        'X_train_xgb': X_train_xgb,
        'X_val_xgb': X_val_xgb,
        'X_test_xgb': X_test_xgb
    }, {
        'num_imputer': num_imputer,
        'cat_imputer': cat_imputer,
        'scaler': scaler,
        'encoder': encoder,
        'input_cols': input_cols,
        'target_col': target_col,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'encoded_cols': encoded_cols
},
        

def preprocess_new_data(data, components):
    #data[components['numeric_cols']] = components['num_imputer'].transform(data[components['numeric_cols']])
    #data[components['numeric_cols']] = components['scaler'].transform(data[components['numeric_cols']])
    #data[components['categorical_cols']] = components['cat_imputer'].transform(data[components['categorical_cols']])
    #data[components['encoded_cols']] = components['encoder'].transform(data[components['categorical_cols']])
    data[components['numeric_cols']] = data[components['numeric_cols']].astype('float64')
    data[components['categorical_cols']] = data[components['categorical_cols']].astype('category')
    X = data[components['numeric_cols'] + components['categorical_cols']]
    data = data.copy()
    return X, data

def split_data(raw_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the raw DataFrame into training and validation sets.
    
    Parameters:
    raw_df (pd.DataFrame): The raw input DataFrame.
    target_col (str): The column name to be used as the target.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: The training and validation DataFrames.
    """
    year = pd.to_datetime(raw_df.Date).dt.year
    train_df = raw_df[year<=2014]
    val_df = raw_df[year==2015]
    test_df = raw_df[year>2015]
    return train_df, val_df, test_df

def del_cols(col_list: List[str], delete_cols: List[str]) -> List[str]:
    """
    Delete specified columns from a list of columns.
    
    Parameters:
    col_list (List[str]): The original list of columns.
    delete_cols (List[str]): List of columns to delete.
    
    Returns:
    List[str]: The updated list of columns.
    """
    for col in delete_cols:
        if col in col_list:
            col_list.remove(col)
    return col_list

def define_columns(df: pd.DataFrame, delete_cols: List[str]) -> Tuple[List[str], str, List[str], List[str]]:
    """
    Define input, target, numeric, and categorical columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    delete_cols (List[str]): List of columns to delete.
    
    Returns:
    Tuple[List[str], str, List[str], List[str]]: Input columns, target column, numeric columns, and categorical columns.
    """
    # Define input, target, numeric, categorical columns
    input_cols = df.columns.tolist()[:-1]
    target_col = df.columns.tolist()[-1]

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:-1]
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Remove specified columns from column lists
    input_cols = del_cols(input_cols, delete_cols)
    numeric_cols = del_cols(numeric_cols, delete_cols)
    categorical_cols = del_cols(categorical_cols, delete_cols)

    return input_cols, target_col, numeric_cols, categorical_cols

def create_inputs_targets(df: pd.DataFrame, input_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create input and target DataFrames from the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    input_cols (List[str]): List of column names to be used as inputs.
    target_col (str): The column name to be used as the target.
    
    Returns:
    Tuple[pd.DataFrame, pd.Series]: The inputs and targets.
    """
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    return inputs, targets

def preprocess_numeric(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, test_inputs, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scale numeric columns using MinMaxScaler.
    
    Parameters:
    train_inputs (pd.DataFrame): Training inputs.
    val_inputs (pd.DataFrame): Validation inputs.
    numeric_cols (List[str]): List of numeric columns.
    scaler (MinMaxScaler): Scaler instance.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]: Scaled training and validation inputs, and the scaler used.
    """
    num_imputer = SimpleImputer(strategy = 'mean')
    num_imputer.fit(train_inputs[numeric_cols])
    train_inputs[numeric_cols] = num_imputer.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = num_imputer.transform(val_inputs[numeric_cols])
    test_inputs[numeric_cols] = num_imputer.transform(test_inputs[numeric_cols])
    scaler = MinMaxScaler()
    scaler.fit(train_inputs[numeric_cols])
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])
    return train_inputs, val_inputs, test_inputs, num_imputer, scaler

def preprocess_categorical(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, test_inputs, categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], OneHotEncoder]:
    """
    Encode categorical columns using OneHotEncoder.
    
    Parameters:
    train_inputs (pd.DataFrame): Training inputs.
    val_inputs (pd.DataFrame): Validation inputs.
    categorical_cols (List[str]): List of categorical columns.
    encoder (OneHotEncoder): Encoder instance.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, List[str], OneHotEncoder]: Encoded training and validation inputs, list of encoded column names, and the encoder used.
    """
    cat_imputer = SimpleImputer(strategy = 'most_frequent')
    cat_imputer.fit(train_inputs[categorical_cols])
    train_inputs[categorical_cols] = cat_imputer.transform(train_inputs[categorical_cols])
    val_inputs[categorical_cols] = cat_imputer.transform(val_inputs[categorical_cols])
    test_inputs[categorical_cols] = cat_imputer.transform(test_inputs[categorical_cols])
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_inputs[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
    test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])
    return train_inputs, val_inputs, test_inputs, cat_imputer, encoder, encoded_cols
