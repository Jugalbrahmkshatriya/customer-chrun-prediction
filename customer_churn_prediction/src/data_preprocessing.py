import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop missing values
    df.dropna(inplace=True)

    # Drop customerID (no predictive value)
    df.drop(columns=['customerID'], inplace=True)

    return df

if __name__ == "__main__":
    df = load_data("data/churn.csv")
    df = clean_data(df)
    print(df.head())
