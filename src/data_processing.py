import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(target_type,data_dir="/kaggle/input/nudat-3/"):

"""
    Loads, cleans, preprocesses, and splits the nuclear data for a specific target.

    Args:
        target_type (str): The target observable. Must be one of 'Sn', 'Sp', or 'S2n'.
        data_dir (str): The directory containing the data files.

    Returns:
        tuple: A tuple containing (X_train_scaled, X_test_scaled, y_train, y_test, scaler).
    """

    if target_type == 'Sn':
        filepath = f"{data_dir}nndc_nudat_data_export_Sn.xlsx"
        dataset = pd.read_excel(filepath)
        target_column = 'Sn'
        Sn_num_missing_values = dataset.isna().sum()
        print(f"number of missing values in Sn_dataset : {Sn_num_missing_values}")
    elif target_type == 'Sp':
        filepath = f"{data_dir}nndc_nudat_data_export_Sp.xlsx"
        dataset = pd.read_excel(filepath)
        df.drop(df.index[[0]], inplace=True)
        target_column = 'Sp'
        Sp_num_missing_values = dataset.isna().sum()
        print(f"number of missing values in Sp_dataset : {Sp_num_missing_values}")
    elif target_type == 'S2n':
        filepath = f"{data_dir}nndc_nudat_data_export_S2n.xlsx"
        dataset = pd.read_excel(filepath)
        target_column = 'S2n'
        S2n_num_missing_values = dataset.isna().sum()
        print(f"number of missing values in S2n_dataset : {S2n_num_missing_values}")
    else:
        raise ValueError("target_type must be one of 'Sn', 'Sp', or 'S2n'")
        
    dataset["n_odd"] = dataset["n"]%2 != 0
    dataset["n_odd"] = dataset["n_odd"].astype(int)
    dataset["z_odd"] = dataset["z"]%2 != 0
    dataset["z_odd"] = dataset["z_odd"].astype(int)
    dataset["n/p"] = dataset["n"]/dataset["z"]
    dataset["A"] = dataset["n"] + dataset["z"]
    dataset["A^2/3"] = dataset["A"]**(2/3)
    dataset["(n-z)/A"] = (dataset["n"] - dataset["z"])/dataset["A"]
    bins = [0, 28, 50, 82, 126, float("inf")]
    labels = [0, 1, 2, 3, 4]
    dataset["z_shell"] = pd.cut(dataset["z"], bins=bins, labels=labels)
    dataset["n_shell"] = pd.cut(dataset["n"], bins=bins, labels=labels)

    Sep_energy = f"{target_column} (KeV)"
    X = dataset.drop(columns=[Sep_energy]).values.astype("float32")
    Y = dataset[Sep_energy].values.astype("float32")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_scaler = StandardScaler().fit(X_train)
    Y_scaler = StandardScaler().fit(Y_train.reshape(-1, 1))
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    Y_train = Y_scaler.transform(Y_train.reshape(-1, 1)).ravel()
    Y_test = Y_scaler.transform(Y_test.reshape(-1, 1)).ravel()

    joblib.dump(Y_scaler, f'{target_type}_scaler.gz')
    print(f"Scaler saved to {target_type}_scaler.gz")

    return X_train , Y_train , X_test , Y_test , Y_scaler
