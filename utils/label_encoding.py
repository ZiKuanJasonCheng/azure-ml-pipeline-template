import pandas as pd

def frequency_encoding(df: pd.DataFrame, columns: list) -> (pd.DataFrame, dict):
    """Frequency encoding function, order by frequency, from 0 to n"""
    
    encoding_mappings = {} # To store encoding mappings

    for col in columns:
        encoding = df[col].value_counts()
        encoding = encoding.sort_values().index.tolist()
    
        # Create dictionary for mapping the categories to the encoding
        ordinal_encoding = {k: i for i, k in enumerate(encoding, 0)}

        # Save the encoding mappings for this column
        encoding_mappings[col] = ordinal_encoding
    
    # Map the ordinal encoding to the dataframe
        df[col] = df[col].map(ordinal_encoding)
    
    return df, encoding_mappings