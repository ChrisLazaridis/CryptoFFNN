def normalization_min_max(df):
    for column in df.columns:
        if column != 'good_buy' and column != 'timestamp':
            min_val = df[column].min()
            max_val = df[column].max()
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df
