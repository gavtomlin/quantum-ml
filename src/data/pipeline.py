import polars as pl 
from sklearn.preprocessing import StandardScaler 
from config import PipelineConfig 

BINARY_COLS = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
SERVICE_COLS = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

BINARY_MAP      = {'Yes': 1, 'No': 0}
SERVICE_MAP     = {'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0}
CONTRACT_MAP    = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
GENDER_MAP      = {'Female': 0, 'Male': 1}
CHURN_MAP       = {'Yes': 1, 'No': 0}

class CleaningPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._scaler = StandardScaler()

    def fit_transform(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
        df = self._drop_id(df)
        df = self._cast_native_numerics(df)
        df = self._fix_total_charges(df)
        df = self._encode_binary(df)
        df = self._encode_service_cols(df)
        df = self._encode_contract(df)  
        df = self._encode_internet_service(df)
        df = self._encode_payment_method(df)
        df = self._encode_gender(df)
        y, df = self._extract_target(df)
        df = self._scale_numeric(df)
        return df, y

    def _drop_id(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop('customerID')

    def _cast_native_numerics(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.col('SeniorCitizen').cast(pl.Int8),
            pl.col('tenure').cast(pl.Int32),
            pl.col('MonthlyCharges').cast(pl.Float64),
            ])

    def _fix_total_charges(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
                pl.col('TotalCharges').str.strip_chars().cast(pl.Float64, strict=False)
        )
        median = df['TotalCharges'].median()
        return df.with_columns(pl.col('TotalCharges').fill_null(median))

    def _encode_binary(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.col(c).replace(BINARY_MAP).cast(pl.Int8) for c in BINARY_COLS
        ])

    def _encode_service_cols(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.col(c).replace(SERVICE_MAP).cast(pl.Int8) for c in SERVICE_COLS
        ])

    def _encode_contract(self, df: pl.DataFrame) -> pl.DataFrame: 
        return df.with_columns(
                pl.col('Contract').replace(CONTRACT_MAP).cast(pl.Int8)
        )

    def _encode_internet_service(self, df: pl.DataFrame) -> pl.DataFrame:
        dummies = df.select('InternetService').to_dummies('InternetService')
        return pl.concat([df.drop('InternetService'), dummies], how='horizontal')

    def _encode_payment_method(self, df: pl.DataFrame) -> pl.DataFrame:
        dummies = df.select('PaymentMethod').to_dummies('PaymentMethod')
        dummies = dummies.drop('PaymentMethod_Bank transfer (automatic)')
        return pl.concat([df.drop('PaymentMethod'), dummies], how='horizontal')

    def _encode_gender(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
                pl.col('gender').replace(GENDER_MAP).cast(pl.Int8)
        )

    def _extract_target(self, df: pl.DataFrame) -> tuple[pl.Series, pl.DataFrame]:
        y = df['Churn'].replace(CHURN_MAP).cast(pl.Int8)
        return y, df.drop('Churn')

    def _scale_numeric(self, df: pl.DataFrame) -> pl.DataFrame:
        scaled = self._scaler.fit_transform(df.select(NUMERIC_COLS).to_numpy())
        scaled_df = pl.DataFrame(scaled, schema={c: pl.Float64 for c in NUMERIC_COLS})
        return pl.concat([df.drop(NUMERIC_COLS), scaled_df], how='horizontal')
