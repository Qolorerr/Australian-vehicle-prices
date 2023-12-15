import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def prepare_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train, df_test = _preprocess_data(df_train=df_train, df_test=df_test)
    print(df_train.info())
    df_train, df_test = _drop_useless_features(df_train=df_train, df_test=df_test)
    print(df_train.info())
    df_train, df_test, y_train, y_test = _fill_nans(
        df_train=df_train, df_test=df_test, y_train=y_train, y_test=y_test
    )
    df_train, df_test = _add_features(df_train=df_train, df_test=df_test)
    df_train, df_test = _normalize_data(df_train=df_train, df_test=df_test)

    return df_train, df_test, y_train, y_test


def logarithmize_y(y: pd.DataFrame) -> pd.DataFrame:
    return y.apply("log")


def expose_y(y: np.ndarray) -> np.ndarray:
    return np.exp(y)


def _preprocess_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for df in (df_train, df_test):
        df.rename(
            columns={
                "Car/Suv": "CarSuv",
            },
            inplace=True,
        )

        df["Year"] = df["Year"].astype(int)
        df["Engine"] = df["Engine"].str.extract(r'(\d+(\.\d+)?)\s*L')[0].astype(np.float64)
        df.rename(columns={"Engine": "EngineVol"}, inplace=True)
        df["FuelConsumption"] = df["FuelConsumption"].str.extract(r'(\d+(\.\d+)?)\s*L\s')[0].astype(np.float64)
        df["Kilometres"] = pd.to_numeric(df["Kilometres"], errors='coerce').astype('Int64')
        df[["ColorExt", "ColorInt"]] = df["ColourExtInt"].str.extract(r'(\w+)\s*/\s*(\w+)')
        df.drop(columns="ColourExtInt", inplace=True)
        df.replace('-', np.nan, inplace=True)
        # df["CylindersinEngine"] = df["CylindersinEngine"].str.extract(r'(\d+)\s').astype('Int64')
        df["Doors"] = df["Doors"].str.extract(r'(\d+)\s')[0].astype('Int64')
        df["Seats"] = df["Seats"].str.extract(r'(\d+)\s')[0].astype('Int64')

        # удаляю кореллирующий признак
        df.drop(columns="CylindersinEngine", inplace=True)

    return df_train, df_test


_NA_GROUP_FILL_NUM_COLS = ["EngineVol", "FuelConsumption", "Kilometres", "Doors", "Seats"]
_NA_GROUP_FILL_CAT_COLS = ["Transmission", "FuelType"]


def _fill_nans(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert not y_train.isna().sum() and not y_test.isna().sum()

    # выбрасываем признаки, процент пропущенных значений у которых больше 80
    nan_info_train = (df_train.isnull().mean() * 100).reset_index()
    nan_info_train.columns = ["column_name", "percentage"]
    nan80_cols = list(nan_info_train[nan_info_train.percentage > 80]["column_name"])
    df_train.drop(columns=nan80_cols, inplace=True)
    df_test.drop(columns=nan80_cols, inplace=True)

    # заполняем модой с учетом группы
    col = "CarSuv"
    fill_value_gen = df_train[col].dropna().mode()[0]
    for body_type, df_group in df_train.groupby("BodyType"):
        mode_output = df_group[col].dropna().mode()
        fill_value = mode_output[0] if len(mode_output) else fill_value_gen
        for df in (df_train, df_test):
            mask = df["BodyType"] == body_type
            df.loc[mask, col] = df.loc[mask, col].fillna(fill_value)
    col = "BodyType"
    fill_value_gen = df_train[col].dropna().mode()[0]
    for car_suv, df_group in df_train.groupby("CarSuv"):
        mode_output = df_group[col].dropna().mode()
        fill_value = mode_output[0] if len(mode_output) else fill_value_gen
        for df in (df_train, df_test):
            mask = df["CarSuv"] == car_suv
            df.loc[mask, col] = df.loc[mask, col].fillna(fill_value)
    for col in _NA_GROUP_FILL_CAT_COLS:
        fill_value_gen = df_train[col].dropna().mode()[0]
        for body_type, df_group in df_train.groupby("BodyType"):
            mode_output = df_group[col].dropna().mode()
            fill_value = mode_output[0] if len(mode_output) else fill_value_gen
            for df in (df_train, df_test):
                mask = df["BodyType"] == body_type
                df.loc[mask, col] = df.loc[mask, col].fillna(fill_value)

    int_columns = df_train.select_dtypes(include=["int32", "Int64", "int64"],
                                         exclude=["object", "float64"]).columns.tolist()

    # заполняем медианой с учетом группы
    for col in _NA_GROUP_FILL_NUM_COLS:
        na_group_fill_num_mapping = (
            df_train.groupby("BodyType")[col].median().to_dict()
        )
        for df in (df_train, df_test):
            for body_type, fill_value in na_group_fill_num_mapping.items():
                mask = df["BodyType"] == body_type
                if col in int_columns:
                    fill_value = int(fill_value)
                df.loc[mask, col] = df.loc[mask, col].fillna(fill_value)
        # print(df.info())

        # заполняем медианой и модой все оставшиеся столбцы с пропущенными значениями
        num_cols = df_train.select_dtypes(exclude=["object"]).columns
        cat_cols = df_train.select_dtypes(include=["object"]).columns
        num_imputer = SimpleImputer(strategy="median")
        cat_imputer = SimpleImputer(strategy="most_frequent")
        num_imputer.fit(df_train[num_cols])
        cat_imputer.fit(df_train[cat_cols])
        for df in (df_train, df_test):
            df[num_cols] = num_imputer.transform(df[num_cols])
            df[cat_cols] = cat_imputer.transform(df[cat_cols])

    return df_train, df_test, y_train, y_test


def _drop_useless_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # удаляю бесполезные признаки
    df_train.drop(columns=["Title", "Location"], inplace=True)
    df_test.drop(columns=["Title", "Location"], inplace=True)

    return df_train, df_test


def _add_ohe_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_cols = df_train.select_dtypes(exclude=["object"]).columns
    cat_cols = df_train.select_dtypes(include=["object"]).columns

    onehot_encoder = OneHotEncoder(
        sparse_output=False, min_frequency=0.3, handle_unknown="ignore"
    ).fit(df_train[cat_cols])
    df_train_ohe = pd.DataFrame(
        onehot_encoder.transform(df_train[cat_cols]),
        columns=onehot_encoder.get_feature_names_out(),
        index=df_train.index,
    )
    df_test_ohe = pd.DataFrame(
        onehot_encoder.transform(df_test[cat_cols]),
        columns=onehot_encoder.get_feature_names_out(),
        index=df_test.index,
    )
    df_train = pd.concat([df_train[num_cols], df_train_ohe], axis=1)
    df_test = pd.concat([df_test[num_cols], df_test_ohe], axis=1)

    return df_train, df_test


def _add_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = _add_ohe_features(df_train=df_train, df_test=df_test)

    return df_train, df_test


def _normalize_data(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = RobustScaler()
    df_train[df_train.columns] = scaler.fit_transform(df_train)
    df_test[df_test.columns] = scaler.transform(df_test)

    return df_train, df_test


def _normalize_data(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = RobustScaler()
    df_train[df_train.columns] = scaler.fit_transform(df_train)
    df_test[df_test.columns] = scaler.transform(df_test)

    return df_train, df_test
