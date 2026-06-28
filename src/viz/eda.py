from pathlib import Path
import polars as pl
import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots

OUTPUT_DIR = Path("outputs")

CATEGORICAL_COLS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod',
        ]

NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

def save(fig, filename: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.write_html(OUTPUT_DIR / filename)

def plot_churn_rate_by_category(df: pl.DataFrame) -> None:
    fig = make_subplots(
            rows=4, cols=4,
            subplot_titles=CATEGORICAL_COLS,
        )
    for i, col in enumerate(CATEGORICAL_COLS): 
        row, col_idx = divmod(i, 4)
        churn_rate = (
                df.group_by([col, 'Churn'])
                .len()
                .with_columns((pl.col('len') / pl.col('len').sum().over(col)).alias('rate')).filter(pl.col('Churn') == 'Yes').sort(col)
                )
        trace = go.Bar(
                x=churn_rate[col].to_list(),
                y=churn_rate['rate'].to_list(),
                name=col,
                showlegend=False,
            )
        fig.add_trace(trace, row=row + 1, col=col_idx + 1)

    fig.update_layout(
            title='Churn Rate by Category',
            height=900,
            template='plotly_dark',
        )
    save(fig, 'churn_rate_by_category.html')

def plot_numeric_distributions(df: pl.DataFrame) -> None:
    fig = make_subplots(rows=1, cols=3, subplot_titles=NUMERIC_COLS)

    for i, col in enumerate(NUMERIC_COLS):
        for churn_val, color in [('Yes', '#EF553B'), ('No', '#636EFA')]:
                values = df.filter(pl.col('Churn') == churn_val).with_columns(
                    pl.col(col).cast(pl.Float64, strict=False)
                )[col].to_list()
                fig.add_trace(
                        go.Histogram(
                            x=values,
                            name=f'Churn={churn_val}',
                            opacity=0.6,
                            marker_color=color,
                            showlegend=(i == 0),
                        ),
                        row=1, col=i + 1,
                    )
    
    fig.update_layout(
        barmode='overlay',
        title='Numeric Feature Distributions by Churn', 
        template='plotly_dark',
        height=400,
    )
    save(fig, 'numeric_distribution.html')

def plot_correlation_heatmap(df: pl.DataFrame) -> None:
    numeric_df = df.select(NUMERIC_COLS).with_columns([
        pl.col(c).cast(pl.Float64, strict=False) for c in NUMERIC_COLS
    ])
    corr = numeric_df.corr()

    fig = px.imshow(
            corr.to_numpy(),
            x=NUMERIC_COLS,
            y=NUMERIC_COLS,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title='Feature Correlation Heatmap', 
            template='plotly_dark',
        )
    save(fig, 'correlation_heatmap.html')

def plot_tenure_vs_charges(df: pl.DataFrame) -> None:
    plot_df = df.with_columns([
        pl.col('tenure').cast(pl.Int32, strict=False),
        pl.col('MonthlyCharges').cast(pl.Float64, strict=False),
    ]).sort('tenure')
    fig = px.scatter(
            x=plot_df['tenure'].to_list(),
            y=plot_df['MonthlyCharges'].to_list(),
            color=plot_df['Churn'].to_list(),
            opacity=0.5,
            title='Tenure vs Monthly Charges by Churn',
            template='plotly_dark',
            color_discrete_map={'Yes': '#EF553B', 'No': '#636EFA'},
            labels={'x': 'tenure', 'y': 'MonthlyCharges', 'color': 'Churn'},
        )
    save(fig, 'tenure_vs_charges.html')

def run_eda(df: pl.DataFrame) -> None:
    plot_churn_rate_by_category(df)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(df)
    plot_tenure_vs_charges(df)
    print("EDA plots saved to outputs/")
