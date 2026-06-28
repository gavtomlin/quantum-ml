from pathlib import Path 
import polars as pl
from schema import RawCustomerRecord

def load_raw(path: str | Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=0)
    _smoke_test(df)
    return df 

def _smoke_test(df: pl.DataFrame) -> None:
    for row in [df.row(0, named=True), df.row(-1, named=True)]:
        RawCustomerRecord(**row)
