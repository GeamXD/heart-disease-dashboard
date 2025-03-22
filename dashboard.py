# Import libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import streamlit as st
import streamlit_shadcn_ui as ui


# Streamlit page name
st.set_page_config(
    page_title="Heart Disease Dashboard",
    page_icon=":heart:",
    layout="wide",
    # initial_sidebar_state="expanded"
)


# Select template
pio.templates.default = "plotly_white"
# 'plotly': Default template.
# 'plotly_white': Clean, white background.
# 'ggplot2': Mimics the style of ggplot2 (R).
# 'seaborn': Mimics the style of Seaborn.
# 'simple_white': Minimalist white background.
# 'presentation': 

# Load dataset
@st.cache_data
def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load dataset from a file
    Params:
        filepath: str - file path
    Returns:
        pd.DataFrame - dataset
    """
    assert filepath.endswith(".csv"), "File must be a CSV file"
    assert isinstance(filepath, str), "File path must be a string"
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Clean dataset
@st.cache_data
def clean_dataset(df: pd.DataFrame, duplicates: bool) -> pd.DataFrame:
    """
    Clean dataset
    Params:
        df: pd.DataFrame - dataset
        duplicates: bool - remove duplicates
    Returns:
        pd.DataFrame - cleaned dataset
    """
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert isinstance(duplicates, bool), "Duplicates must be a boolean"

    # Fix resting column
    df = df.rename(columns={'resting bp s': 'resting blood pressure'})
    # Recode sex columns
    df['sex'] = df['sex'].replace({
        1: 'Male',
        0: 'Female'
    })

    # Recode chest pain type columns
    df['chest pain type'] = df['chest pain type'].replace({
        1: 'Typical angina',
        2: 'Atypical angina',
        3: 'Non-anginal pain',
        4: 'Asymptomatic'
    })

    # Recode fasting blood sugar columns
    df['fasting blood sugar'] = df['fasting blood sugar'].replace({
        1: '> 120 mg/dl',
        0: '< 120 mg/dl'
    })

    # Recode resting ecg olumns
    df['resting ecg'] = df['resting ecg'].replace({
        0: 'Normal',
        1: 'ST-T wave abnormality',
        2: 'Left ventricular hypertrophy'
    })

    # Recode exercise angina columns
    df['exercise angina'] = df['exercise angina'].replace({
        1: 'Yes',
        0: 'No'
    })

    # Recode st slope columns
    df['ST slope'] = df['ST slope'].replace({
        1: 'Upsloping',
        2: 'Flat',
        3: 'Downsloping',
        0: 'Flat'
    })


    # Recode target column
    df['target'] = df['target'].replace({
        0: 'No',
        1: 'Yes'
    })
    df = df.rename(columns={'target': 'disease status'})

    # Create age group column
    df['age group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 70, 80], 
                             labels=['0-29', '30-39', '40-49', '50-59', '60-69', '70-79'])

    # Create cholesterol group column
    df['cholesterol group'] = pd.cut(df['cholesterol'], bins=[0, 150, 200, 240, 300, 1000],
                                 labels=['<150 mg/dL', '150-199 mg/dL', '200-239 mg/dL', '240-299 mg/dL', '≥300 mg/dL'])

    if duplicates:
        df.drop_duplicates(inplace=True)
        return df
    else:
        return df

def plot_hist(df: str, xcol: str, title: str, x_title: str, bins: int = 20) -> None:
    """
    Plots a histogram
    Params:
        df: DataFrame
        xcol: str
        title: str
        x_title: str
    Returns:
        None
    """
    fig  = px.histogram(df, x=xcol, title=title, nbins=bins)
    fig.update_layout(
        bargap=0.01,
        width=800,
        height=500,
        yaxis_title='Frequency',
        xaxis_title=x_title,
        title_x=0.5
    )

    return fig


def plot_pie_chart(df: pd.DataFrame, xcol: str, title: str) -> None:
    """
    Plots a pie chart
    Params:
        df: DataFrame
        xcol: str
        title: str
    Returns:
        None
    """
    fig = px.pie(data_frame=df, names=xcol, title=title,  hole=0.6, width=600, height=400)
    fig.update_traces(
    textinfo='label + percent',
    textposition='outside',
    )

    fig.update_layout(
    showlegend=False,
    title_x=0.2,
    )

    return fig

def plot_one_var_bar(df: pd.DataFrame, xcol: str, title: str, x_title: str) -> None:
    """
    Plots a bar chart
    Params:
        df: DataFrame
        xcol: str
        title: str
        x_title: str
    Returns:
        None
    """
    # Aggregate data
    df = df[xcol].value_counts().reset_index()
    fig = px.bar(data_frame=df, x=xcol, y='count', title=title, text_auto=True)

    fig.update_layout(
        width=900,
        height=500,
        xaxis_title=x_title,
        title_x=0.5,
    )

    fig.update_traces(
        textposition='outside',
        )

    return fig

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table
    Params:
        df: DataFrame
    Returns:
        DataFrame
    """
    # Create a summary table
    df = df.describe().round(2).T
    df.drop(['count', '25%', '75%'], axis=1, inplace=True)
    df.rename(columns={'50%': 'median'}, inplace=True)
    return df

def count_contents(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Count contents in a column
    Params:
        df: DataFrame
        col: str
    Returns:
        DataFrame
    """
    return df[col].value_counts().reset_index()

def get_metrics_card_info(df: pd.DataFrame) -> int:
    """
    Get info for metrics card
    Params:
        df: DataFrame
    Returns:
        str
    """
    no_participants = df.shape[0]
    num_disease = str(df['disease status'].value_counts().values[0])
    max_hr = df['max heart rate'].max()
    num_exerc_angina = str(df['exercise angina'].value_counts().values[1])
    return no_participants, num_disease, max_hr, num_exerc_angina

def aggregate_columns(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Aggregates columns
    Params:
        df: DataFrame
        group_col: str
    Returns:
        DataFrame
    """
    df = df.groupby(group_col).agg(
            MinRBP=('resting blood pressure', 'min'),
            MaxRBP=('resting blood pressure', 'max'),
            MeanRBP=('resting blood pressure', 'mean'),
            MinMHR=('max heart rate', 'min'),
            MaxMHR=('max heart rate', 'max'),
            MeanMHR=('max heart rate', 'mean'),
            MinChol=('cholesterol', 'min'),
            MaxChol=('cholesterol', 'max'),
            MeanChol=('cholesterol', 'mean')).reset_index().round(2)
    return df

def plot_multivar_bar(df: pd.DataFrame, xcol: str, color: str, title: str, x_title: str) -> None:
    """
    Plots a bar chart
    Params:
        df: DataFrame
        xcol: str
        title: str
        x_title: str
    Returns:
        None
    """
    # Group by mean
    df = df.groupby([xcol, color]).size().reset_index(name='count')
    fig = px.bar(data_frame=df, x=xcol, y='count', color=color,
                 title=title, text_auto=True, barmode='group')

    fig.update_layout(
        width=900,
        height=500,
        xaxis_title=x_title,
        title_x=0.5,
    )

    fig.update_traces(
        textposition='outside',
        )

    return fig


# Define filepath
filepath = "heart_statlog_cleveland_hungary_final.csv"

# Load dataset
df = load_dataset(filepath)

# Clean dataset
df = clean_dataset(df, duplicates=False)

# Get metrics card info
no_participants, num_disease, max_hr, num_exerc_angina = get_metrics_card_info(df)

# Get pie charts
gender_pie = plot_pie_chart(df, 'sex', 'Gender')
disease_status_pie = plot_pie_chart(df, 'disease status', 'Disease Status')
exercise_angina_pie = plot_pie_chart(df, 'exercise angina', 'Exercise Angina')
fbs_pie = plot_pie_chart(df, 'fasting blood sugar', 'Fasting Blood Sugar')


# Get histogram
max_hr_hist = plot_hist(df, 'max heart rate', 'Max Heart Rate Distribution', 'Max Heart Rate', bins=20)
rest_bp_hist = plot_hist(df, 'resting blood pressure', 'Resting Blood Pressure Distribution', 'Resting Blood Pressure', bins=20)
cholesterol_hist = plot_hist(df, 'cholesterol', 'Cholesterol Distribution', 'Cholesterol', bins=20)
age_hist = plot_hist(df, 'age', 'Age Distribution', 'Age', bins=20)

# Get bar charts
# Simple
chest_paintype_bar = plot_one_var_bar(df, 'chest pain type', 'Chest Pain Type', 'Chest Pain Type')
rest_ecg_bar = plot_one_var_bar(df, 'resting ecg', 'Resting ECG', 'Resting ECG')
st_slope_bar = plot_one_var_bar(df, 'ST slope', 'ST Slope', 'ST Slope')
age_group_bar = plot_one_var_bar(df, 'age group', 'Age Group', 'Age Group')
# cholesterol_group_bar = plot_one_var_bar(df, 'cholesterol group', 'Cholesterol Group', 'Cholesterol Group')

# complex
age_group_by_fbs = plot_multivar_bar(df, 'age group', 'fasting blood sugar', 'Age Group by fasting blood sugar', 'Age Group')
age_group_by_resting_ecg = plot_multivar_bar(df, 'age group', 'resting ecg', 'Age Group by Resting ECG', 'Age Group')
age_group_by_chest_pain = plot_multivar_bar(df, 'age group', 'chest pain type', 'Age Group by Chest Pain Type', 'Age Group')
age_group_by_st_slope = plot_multivar_bar(df, 'age group', 'ST slope', 'Age Group by ST Slope', 'Age Group')
age_group_by_disease_status = plot_multivar_bar(df, 'age group', 'disease status', 'Age Group by Disease Status', 'Age Group')

# Get summary table
sum_stat = summary_table(df)

# Dashboard title
st.markdown("<h1 style='text-align: center;'>Heart Disease Dashboard ♥️</h1>", unsafe_allow_html=True)
st.write("") # White space

# Metrics cards
metric_col = st.columns(4)
with metric_col[0]:
    ui.metric_card(title="Participants", content=no_participants, description="Total", key='card1')
with metric_col[1]:
    ui.metric_card(title="Disease Status", content=num_disease, description="Total", key='card2')
with metric_col[2]:
    ui.metric_card(title="Max Heart Rate", content=max_hr, description="bpm", key='card3')
with metric_col[3]:
    ui.metric_card(title="Exercise Angina", content=num_exerc_angina, description="Total", key='card4')

# Separator
# st.markdown("---")

# Pie charts
pie_cols = st.columns(3)
with pie_cols[0]:
    pi_container = st.container(border=True)
    pi_container.plotly_chart(gender_pie, use_container_width=True)
with pie_cols[1]:
    pi_container2 = st.container(border=True)
    pi_container2.plotly_chart(fbs_pie, use_container_width=True)
with pie_cols[2]:
    pi_container3 = st.container(border=True)
    pi_container3.plotly_chart(disease_status_pie, use_container_width=True)

# Separator
# st.markdown("---")


# Histograms
hist_cols = st.columns(2)
with hist_cols[0]:
    hist_con = st.container(border=True)
    hist_con.plotly_chart(max_hr_hist, use_container_width=True)
with hist_cols[1]:
    hist_con2 = st.container(border=True)
    hist_con2.plotly_chart(age_hist, use_container_width=True)

# Separator
# st.markdown("---")

    
# bar charts - SIMPLE
bar_cols = st.columns(2)
with bar_cols[0]:
    bar_con = st.container(border=True)
    bar_con.plotly_chart(chest_paintype_bar, use_container_width=True)
with bar_cols[1]:
    bar_con2 = st.container(border=True)
    bar_con2.plotly_chart(rest_ecg_bar, use_container_width=True)

# Separator
# st.markdown("---")


# Bar charts with table
bar_cols2 = st.columns(2)
with bar_cols2[0]:
    br_con = st.container(border=True)
    br_con.plotly_chart(age_group_bar, use_container_width=True)
with bar_cols2[1]:
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.subheader('Summary Table')
    st.dataframe(sum_stat, use_container_width=True)


# Separator
st.markdown("---")


# Bar charts - complex
bar_cols = st.columns(2)
with bar_cols[0]:
    comp_con = st.container(border=True)
    comp_con.plotly_chart(age_group_by_disease_status, use_container_width=True)
with bar_cols[1]:
    comp_con1 = st.container(border=True)
    comp_con1.plotly_chart(age_group_by_st_slope, use_container_width=True)

# Separator
st.markdown("---")


# # Tables = modify with filter
var_to_select = ['Sex', 'Chest Pain Type', 'Fasting Blood Sugar', 'Resting ECG', 
                 'Exercise Angina', 'ST slope', 'Disease Status', 'Age Group', 'Cholesterol Group']
st.write("### Filter Data")
agg_col = st.columns(2)
with agg_col[0]:    
    # Filter dropdown
    group_col = st.selectbox(
        "Select a column to aggregate by:",
        options=var_to_select,
        index=0
    )

    # Aggregate the data based on the selected column
    aggregated_df = aggregate_columns(df, group_col.lower())

    # Display the aggregated DataFrame
    st.write("### Aggregated Data")
    st.dataframe(aggregated_df, use_container_width=True)
with agg_col[1]:
    var_to_select1 = ['Age Group', 'Age', 'Chest Pain Type', 'Fasting Blood Sugar', 'Resting ECG', 
                      'Resting Blood Pressure', 'Cholesterol','Exercise Angina', 'ST slope', 'Disease Status',
                      'Sex', 'Cholesterol Group','Max Heart Rate', 'oldpeak']
    # Filter dropdown
    group_col1 = st.selectbox(
        "Select a column:",
        options=var_to_select1,
        index=0
    )
    # Count the data based on the selected column
    count_df = count_contents(df, group_col1.lower())

    # Display the aggregated DataFrame
    st.write("### Aggregated Data")
    st.dataframe(count_df, use_container_width=True)
