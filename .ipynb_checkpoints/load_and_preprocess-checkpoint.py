import joblib

def load_model_components(model_path):
    loaded_model = joblib.load(model_path)
    return {
        'imputer' : loaded_model['imputer'],
        'scaler' : loaded_model['scaler'],
        'encoder' : loaded_model['encoder'],
        'model' : loaded_model['model'],
        'input_cols' : loaded_model['input_cols'],
        'target_col' : loaded_model['target_col'],
        'numeric_cols' : loaded_model['numeric_cols'],
        'categorical_cols' : loaded_model['categorical_cols'],
        'encoded_cols' : loaded_model['encoded_cols']}

def preprocess_data(data, components):
    data[components['numeric_cols']] = components['imputer'].transform(data[components['numeric_cols']])
    data[components['numeric_cols']] = components['scaler'].transform(data[components['numeric_cols']])
    data[components['encoded_cols']] = components['encoder'].transform(data[components['categorical_cols']])
    X = data[components['numeric_cols'] + components['encoded_cols']]
    #X = data[components['numeric_cols']]
    return X, data