def preprocess_data(df, output_file):
    # Drop unnecessary columns
    df.drop(['close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
             'taker_buy_quote_asset_volume', 'ignore'], axis=1, inplace=True)
    # Convert data types
    df = df.astype(float)
    # Handle missing values
    df.dropna(inplace=True)

    # Save data to CSV
    df.to_csv(output_file)
