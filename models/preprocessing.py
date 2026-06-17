from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

def get_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ])

def encode_df(df):
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    return df_enc