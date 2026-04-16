# src/data/preprocess.py

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw credit card fraud dataset."""
    df = pd.read_csv(filepath)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    after = df.shape[0]
    print(f"Removed duplicates: {before - after}")
    return df


def split_features_target(df: pd.DataFrame):
    """Split dataset into features and target."""
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def stratified_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Perform stratified train-test split."""
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def scale_time_amount(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale only Time and Amount using StandardScaler fit on training data."""
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    return X_train, X_test, scaler


def save_processed_data(X_train, X_test, y_train, y_test, scaler, output_dir: str):
    """Save processed train/test sets and scaler."""
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    print(f"Processed files saved to: {output_dir}")


def main():
    raw_path = os.path.join("data", "raw", "creditcard.csv")
    output_dir = os.path.join("data", "processed")

    print("Loading raw data...")
    df = load_data(raw_path)
    print("Raw shape:", df.shape)

    print("\nRemoving duplicates...")
    df = remove_duplicates(df)
    print("Shape after duplicate removal:", df.shape)

    print("\nSplitting features and target...")
    X, y = split_features_target(df)

    print("\nPerforming stratified train/test split...")
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train fraud count:", y_train.sum())
    print("y_test fraud count:", y_test.sum())

    print("\nScaling Time and Amount...")
    X_train, X_test, scaler = scale_time_amount(X_train, X_test)

    print("\nSaving processed files...")
    save_processed_data(X_train, X_test, y_train, y_test, scaler, output_dir)

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()