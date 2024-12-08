#Import All the required libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import requests
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot
import warnings
warnings.filterwarnings("ignore")

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# API Key and Base URL
API_KEY = '2fF4tOa23lFgyTHUMGhhUbb5BPYEDSREghq6LtYv'
BASE_URL = "https://api.eia.gov/v2/electricity/electric-power-operational-data/data/"

# List of U.S. state codes and their full names
state_mapping = {
    "ALL": "All States", "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", 
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", 
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", 
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", 
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", 
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", 
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", 
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", 
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", 
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"
}

# Energy types
energy_types = [
    "NG - Natural Gas", "BIT - Bituminous Coal", "SUB - Subbituminous Coal", "LIG - Lignite Coal", 
    "PC - Petroleum Coke", "DFO - Distillate Fuel Oil", "KER - Kerosene", "RFO - Residual Fuel Oil", 
    "WND - Wind", "NUC - Nuclear", "SUN - Solar", "HYD - Hydropower", "GEO - Geothermal", "OTH - Other"
]

# Reverse mapping for backend processing
state_codes = list(state_mapping.keys())
state_names = list(state_mapping.values())

# Page configuration
st.set_page_config(page_title="Energy Consumption Forecasting", layout="wide")


#-----------------------------------------------------------------------------------------------------------------------------------------------#


# Custom styles
def apply_custom_styles():
    st.markdown("""
        <style>
        .stApp {
            background-color: #FFFFFF; 
            color: black;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .css-1lcbmhc.e1fqkh3o1 {
            background-color: #FFFFFF !important; 
        }
        .css-16huue1.e1fqkh3o3 {
            color: black !important;
        }
        
        h1, h2, h3, h4 {
            color: #003366;  /* Navy blue for headings */
            font-weight: 600;
        }

        button[kind="primary"] {
            background-color: #003366 !important; /* Navy blue button */
            color: white !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            padding: 12px 40px;
            text-align: center;
            transition: background-color 0.3s ease;
        }

        button[kind="primary"]:hover {
            background-color: #00509E !important; /* Lighter blue on hover */
            color: white !important;
        }

        .stButton > button {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }

        /* Sidebar styling */
        .stSidebar {
            background-color: #FFFFFF !important;  /* White background */
            color: black !important;
        }

        .stSelectbox, .stRadio, .stTextInput, .stMultiSelect {
            color: black !important;
            font-weight: 500;
        }

        .stSelectbox label, .stRadio label, .stTextInput label {
            color: #003366 !important;  /* Navy blue for labels */
        }

        /* Radio button label styling */
        .stRadio label {
            color: #003366 !important;  /* Navy blue for radio button labels */
            font-weight: 600 !important;  /* Bold font weight */
        }

        /* Styling for dropdown/selectbox with blue border */
        .stSelectbox select, .stMultiSelect select {
            border: 2px solid #003366 !important;  /* Blue border */
            border-radius: 4px !important;
            padding: 8px 12px;
        }

        .stSelectbox select:focus, .stMultiSelect select:focus {
            border-color: #00509E !important;  /* Lighter blue border on focus */
        }

        /* Table styling */
        .stDataFrame table {
            overflow-y: auto;
            max-height: 400px !important; /* Fixed table height */
            table-layout: auto;
            width: 100% !important;
        }

        .stDataFrame thead th:first-child, .stDataFrame tbody td:first-child {
            display: none;
        }

        .stDataFrame th {
            font-size: 14px;
            background-color: #F0F0F0; /* Light gray background for table headers */
            text-align: center;
        }

        .stDataFrame td {
            font-size: 14px;
            text-align: center;
        }

        .stRadio div {
            color: black !important;
        }

        /* Top bar background color */
        .css-1v3fvcr {
            background-color: #F0F0F0 !important; /* Light gray top bar */
            color: black !important;
        }

        /* Styling for Selectbox label to ensure it appears in navy blue */
        .stSelectbox label {
            color: #003366 !important;  /* Navy blue */
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_styles()


#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Title of the app
st.markdown("<h1 style='text-align: center;'>🔋 Energy Consumption Forecasting</h1>", unsafe_allow_html=True)

# Dropdown for selecting a U.S. state
selected_state_name = st.sidebar.selectbox("Select a U.S. state", state_names)
selected_state_code = state_codes[state_names.index(selected_state_name)]

# Dropdown for selecting energy type
selected_energy_type = st.sidebar.selectbox("Select Energy Type", energy_types)

# Sidebar selection for forecast type
forecast_type = st.sidebar.radio(
    "Select Forecast Type",
    options=["Short-Term (6 months)", "Long-Term (24 months)"],
    index=0
)


#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Function to fetch data from the EIA API
def fetch_energy_data(state_code, energy_type):
    fuel_type = energy_type.split(" - ")[0]
    params = {
        "api_key": API_KEY,
        "frequency": "monthly",
        "data[0]": "total-consumption",
        "facets[fueltypeid][]": fuel_type,
        "facets[sectorid][]": "98",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
        "start": "2014-01",
        "end": "2024-08",
    }
    if state_code != "ALL":
        params["facets[location][]"] = state_code
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'response' in data and 'data' in data['response']:
            df = pd.DataFrame(data['response']['data'])
            return df
        else:
            return None
    else:
        st.error(f"Failed to fetch data. Status code: {response.status_code}")
        return None


#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Fetch data and display it when both selections are made
if st.sidebar.button("Request Forecast"):
    with st.spinner("Fetching data..."):
        data = fetch_energy_data(selected_state_code, selected_energy_type)
        if data is not None and not data.empty:    
            data.rename(columns={'period': 'Period', 'total-consumption': 'Consumption'}, inplace=True)
            data['Period'] = pd.to_datetime(data['Period'], errors='coerce')
            data['Consumption'] = pd.to_numeric(data['Consumption'], errors='coerce')
            data.dropna(subset=['Period', 'Consumption'], inplace=True)
            data = data.set_index('Period').resample('ME').agg({
                "Consumption": "sum",  # Average for numeric column
                "fueltypeid": "first",         # Arbitrary choice for non-numeric columns
                "sectorid": "first",         # Arbitrary choice for non-numeric columns
                "location": "first",
                "sectorDescription": "first",
                "fuelTypeDescription": "first",
                "total-consumption-units": "first",
                "stateDescription": "first"
            })
            data.rename(columns={
                "Period": "Period",
                "Consumption": "Total Consumption",
                "fueltypeid": "Fuel Type",
                "sectorid": "Sector ID",
                "location": "Location",
                "sectorDescription": "Sector Description",
                "fuelTypeDescription": "Fuel Type Description",
                "total-consumption-units": "Consumption Units",
                "stateDescription": "State"
            }, inplace=True)

            def plot_analysis(data):
                # Continuous Moments (First to Fourth)
                st.markdown("<h3 style='text-align: center;'>Continuous Moments (Mean, Variance, Skewness, Kurtosis)</h3>", unsafe_allow_html=True)
                st.write(f"Mean: {data.mean()}")
                st.write(f"Variance: {data.var()}")
                st.write(f"Skewness: {skew(data)}")
                st.write(f"Kurtosis: {kurtosis(data)}")

                # Create a 1x3 subplot layout
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))

                # Histogram
                sns.histplot(data["Total Consumption"], bins=30, kde=True, ax=axs[0], color="blue")
                axs[0].set_title('Histogram of Consumption')

                # Boxplot
                sns.boxplot(x=data["Total Consumption"], ax=axs[1], color="green")
                axs[1].set_title('Boxplot of Consumption')

                # Adjust layout and display the plots
                plt.tight_layout()
                st.pyplot(fig)

                # Autocorrelation and Partial Autocorrelation Plots
                st.markdown("<h3 style='text-align: center;'>ACF and PACF Plots</h3>", unsafe_allow_html=True)

                # Set up the figure and subplots
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))

                # Lag Plot
                lag_plot(data, ax=axs[0])
                axs[0].set_title('Lag Plot of Consumption')

                # ACF Plot
                plot_acf(data, lags=20, ax=axs[1])
                axs[1].set_title('ACF Plot of Consumption')

                # PACF Plot
                plot_pacf(data, lags=20, ax=axs[2])
                axs[2].set_title('PACF Plot of Consumption')

                # Adjust layout and display the plots
                plt.tight_layout()
                st.pyplot(fig)

                # K-means Clustering and Linear Regression
                st.markdown("<h3 style='text-align: center;'>K-means Clustering and Regression</h3>", unsafe_allow_html=True)

                # Ensure data is clean and valid
                data.dropna(subset=['Total Consumption'], inplace=True)

                # Simple Linear Regression
                X = data.index.map(pd.Timestamp.toordinal)  # Convert the index (Period) to ordinal for regression
                X = sm.add_constant(X)  # Add constant (intercept) to the model
                y = data['Total Consumption']  # Dependent variable (Total Consumption)
                model = sm.OLS(y, X).fit()
                preds = model.predict(X)
                print(model.summary())

            ##Function to call EDA
            df_EDA = data.copy()
            df_EDA = df_EDA[['Total Consumption']]  # Ensure only the numeric column is selected
            df_EDA['Total Consumption'] = pd.to_numeric(df_EDA['Total Consumption'], errors='coerce')  # Convert to numeric
            #plot_analysis(df_EDA) #Uncomment this line to see EDA Plots on the Streamlit WebApplication
            

#-----------------------------------------------------------------------------------------------------------------------------------------------#

            st.markdown(f"<h4 style='text-align: center;'>{'Energy Consumption Data for ' + selected_state_name + ' (' + selected_energy_type + ')'}</h4>", unsafe_allow_html=True)
            st.dataframe(data.style.format({"Total Consumption": "{:,.2f}"}))
            data_Copy = data.copy()

            # Top 5 highest consumption
            top_5_data = data.nlargest(5, 'Total Consumption')  # Select top 5 highest consumptions

            # Display the top 5 highest consumption as a table
            st.markdown("<h4 style='text-align: center;'>Top 5 Highest Energy Consumptions -  Monthly</h4>", unsafe_allow_html=True)
            st.dataframe(top_5_data[['Total Consumption', 'Consumption Units']].style.format({"Total Consumption": "{:,.2f}"}))

#-----------------------------------------------------------------------------------------------------------------------------------------------#
            # Generate dynamic interpretation
            st.markdown("<h4 style='text-align: center;'>Line Plot of Average Monthly Consumption</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            data.groupby('Period')['Total Consumption'].mean().plot(ax=ax)
            #ax.set_title('Line Plot of Average Monthly Consumption')
            st.pyplot(fig)

            # Filter data based on forecast type
            if forecast_type == "Short-Term (6 months)":
                forecast_horizon = 6
            else:  # Long-Term (24 months)
                forecast_horizon = 24

#-----------------------------------------------------------------------------------------------------------------------------------------------#
            # Data preprocessing
            data = data[['Total Consumption']]  # Ensure only the numeric column is selected
            data['Total Consumption'] = pd.to_numeric(data['Total Consumption'], errors='coerce')  # Convert to numeric
            data = data.dropna()

            # Train-test split
            train_size = int(len(data) * 0.7)
            train, test = data[:train_size], data[train_size:]
            test_len = len(test)

#-----------------------------------------------------------------------------------------------------------------------------------------------#
            # Function Definitions
            def calculate_metrics(test, forecast):
                return {
                    "RMSE": np.sqrt(mean_squared_error(test, forecast[:len(test)])),
                    "MAPE": mean_absolute_percentage_error(test, forecast[:len(test)])
                }
            def train_arima(train, test, forecast_horizon):
                model = auto_arima(train,
                                    start_p=1,  # Minimum AR terms to consider
                                    max_p=3,  # Maximum AR terms to consider based on PACF
                                    start_q=1,  # Minimum MA terms to consider
                                    max_q=3,  # Maximum MA terms to consider based on ACF
                                    d=1,  # Non-seasonal differencing (to handle trends)
                                    start_P=0,  # Minimum seasonal AR terms
                                    max_P=2,  # Maximum seasonal AR terms
                                    start_Q=0,  # Minimum seasonal MA terms
                                    max_Q=2,  # Maximum seasonal MA terms
                                    D=1,  # Seasonal differencing (to handle seasonality)
                                    seasonal=True,
                                    m=12,
                                    trace=True,
                                    random_state=42,
                                    suppress_warnings=True,
                                    stepwise=True)
                fitted_model = model.fit(train)
                forecast = model.predict(n_periods=forecast_horizon)
                return forecast, fitted_model

            def train_lstm(train, test, forecast_horizon):
                scaler = MinMaxScaler()
                scaled_train = scaler.fit_transform(train.values.reshape(-1, 1))
                generator = TimeseriesGenerator(scaled_train, scaled_train, length=12, batch_size=1)

                model = Sequential([
                    LSTM(50, activation='relu', input_shape=(12, 1)),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(generator, epochs=20, verbose=0)

                lstm_predictions = []
                batch = scaled_train[-12:]
                current_batch = batch.reshape((1, 12, 1))
                for _ in range(forecast_horizon):
                    lstm_pred = model.predict(current_batch)[0]
                    lstm_predictions.append(lstm_pred)
                    current_batch = np.append(current_batch[:, 1:, :], [[lstm_pred]], axis=1)

                lstm_predictions = scaler.inverse_transform(lstm_predictions)
                return lstm_predictions.flatten(), model

            def train_naive(train, test, forecast_horizon):
                forecast = np.full(forecast_horizon, train.iloc[-1])
                return forecast, None

            def train_exponential_smoothing(train, test, forecast_horizon):
                model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
                # Fit the model with custom smoothing parameters
                fitted_ets_model = model.fit(
                smoothing_level=0.2,   # alpha
                smoothing_slope=0.1,   # beta
                smoothing_seasonal=0.3  # gamma
                )
                forecast = fitted_ets_model.forecast(forecast_horizon)
                return forecast, fitted_ets_model

            # Calculate Metrics
            def calculate_metrics(test, forecast):
                return {
                    "RMSE": np.sqrt(mean_squared_error(test, forecast[:len(test)])),
                    "MAPE": mean_absolute_percentage_error(test, forecast[:len(test)])
                }

#-----------------------------------------------------------------------------------------------------------------------------------------------#
            # Train all models
            with st.spinner("Training models...(Might take 3-5 minutes)"):
                arima_forecast, arima_model = train_arima(train, test, test_len)
                lstm_forecast, lstm_model = train_lstm(train, test, test_len)
                naive_forecast, naive_model = train_naive(train, test, test_len)
                es_forecast, es_model = train_exponential_smoothing(train, test, test_len)

            # Calculate metrics
            arima_metrics = calculate_metrics(test, arima_forecast)
            lstm_metrics = calculate_metrics(test, lstm_forecast)
            naive_metrics = calculate_metrics(test, naive_forecast)
            es_metrics = calculate_metrics(test, es_forecast)

            # Select the best model
            metrics_df = pd.DataFrame({
                "Model": ["ARIMA", "LSTM", "Naive", "Exponential Smoothing"],
                "RMSE": [arima_metrics["RMSE"], lstm_metrics["RMSE"], naive_metrics["RMSE"], es_metrics["RMSE"]],
                "MAPE": [arima_metrics["MAPE"], lstm_metrics["MAPE"], naive_metrics["MAPE"], es_metrics["MAPE"]]
            })
            best_model = metrics_df.loc[metrics_df["MAPE"].idxmin(), "Model"]

            # Display best model and metrics
            st.markdown("<h4>Model Performance Metrics</h4>", unsafe_allow_html=True)
            # Center-align the dataframe
            st.markdown("<div style='display: flex; justify-content: center; width: 100%;'>", unsafe_allow_html=True)
            st.dataframe(metrics_df.style.format({"RMSE": "{:.2f}", "MAPE": "{:.2%}"}))
            st.markdown("</div>", unsafe_allow_html=True)

            # Best Model
            st.markdown(f"<h4 style='text-align: center;'>Best Model: {best_model}</h4>", unsafe_allow_html=True)

            # Generate forecast using the best model
            if best_model == "ARIMA":
                forecast = arima_forecast
            elif best_model == "LSTM":
                forecast = lstm_forecast
            elif best_model == "Naive":
                forecast = naive_forecast
            else:  # Exponential Smoothing
                forecast = es_forecast

            # Define forecast period based on the forecast type
            forecast_period = pd.date_range(start='2024-09', periods=forecast_horizon, freq='M')

            # Function to Generate Forecast and Confidence Intervals
            def get_forecast_and_confidence(model_name, forecast_horizon):
                if model_name == "ARIMA":
                    forecast = arima_model.predict(n_periods=forecast_horizon)  
                    conf_int = arima_model.conf_int(alpha=0.05)  # Confidence interval for ARIMA
                    return forecast, conf_int
                elif model_name == "LSTM":
                    
                    forecast = lstm_forecast[:forecast_horizon]
                    return forecast, None  # No confidence intervals for LSTM
                elif model_name == "Naive":
                    forecast = np.full(forecast_horizon, train.iloc[-1])
                    return forecast, None  # No confidence intervals for Naive
                elif model_name == "Exponential Smoothing":
                    forecast = es_model.forecast(forecast_horizon)
                    return forecast, None  # No confidence intervals for Exponential Smoothing
                else:
                    raise ValueError("Invalid Model Name")

            # Generate Forecast and Confidence Intervals
            forecast, conf_int = get_forecast_and_confidence(best_model, forecast_horizon)
            # Plot Historical Data, Forecast, and Confidence Intervals
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(data, label='Historical Data', color='blue', linewidth=2)
            ax.plot(forecast_period, forecast, label=f'Forecast ({best_model})', color='red', linewidth=2)

            # Add Confidence Interval for ARIMA
            if best_model == "ARIMA" and conf_int is not None:
                conf_int = np.array(conf_int)
                #ax.fill_between(forecast_period, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label="95% Confidence Interval")

            # Customize Plot
            ax.set_title(f"{forecast_type} {best_model} Forecast with Confidence Interval", fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Consumption", fontsize=12)
            ax.legend()
            plt.grid()
            st.pyplot(fig)

            # Display Forecasted Values in a Table
            unit = data_Copy["Consumption Units"].iloc[0]
            forecast_df = pd.DataFrame({
                "Date": forecast_period,
                "Forecast": forecast,
                "Units" : unit
            })

#-----------------------------------------------------------------------------------------------------------------------------------------------#
            # If ARIMA has confidence intervals, add them to the dataframe
            #if best_model == "ARIMA" and conf_int is not None:
               # forecast_df["Lower Bound (95%)"] = conf_int[:, 0]
              #  forecast_df["Upper Bound (95%)"] = conf_int[:, 1]

            # Display the forecasted values table
            st.markdown(f"<h4>{forecast_type} Forecasted Values</h4>", unsafe_allow_html=True)
            st.dataframe(forecast_df)

            # Forecast Period for plotting
            forecast_index = pd.date_range(start=data.index[-1], periods=forecast_horizon + 1, freq='M')[1:]

            # Plot Forecast with Confidence Interval for ARIMA
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_title(f"{forecast_type} {best_model} Forecast with Historical Data", fontsize=16)
            ax.plot(data, label='Historical Data')
            ax.plot(forecast_index, forecast, label='Forecast', color='red')
            if best_model == "ARIMA":
                conf_int = arima_model.conf_int(alpha=0.05)
                #ax.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
            ax.legend()
            st.pyplot(fig)

            # Display the forecast values in a table for the forecast period
            forecast_df = pd.DataFrame({"Date": forecast_index, "Forecast": forecast,"Units":unit})
            st.markdown(f"<h4>{forecast_type} Forecasted Values</h4>", unsafe_allow_html=True)
            st.dataframe(forecast_df)
        else:
            st.warning(f"No data available for {selected_state_name} ({selected_energy_type}).")



