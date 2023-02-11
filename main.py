# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px

# Define start date and today's date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Title of the app
st.title('Stock Forecast App')

# List of stocks to choose from
stocks = ('AMZN', 'NFLX', 'TSLA', 'JPM', 'JNJ', 'BRK.A','GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

time_based_forecast = st.selectbox('Select time-based forecasting:', ('Next day', 'Next week', 'Next month'))

# Slider to select number of years for prediction
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


# Load data for the selected stock using yfinance library
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')


st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
if time_based_forecast == 'Next day':
    future = m.make_future_dataframe(periods=1)
elif time_based_forecast == 'Next week':
    future = m.make_future_dataframe(periods=7)
else:
    future = m.make_future_dataframe(periods=30)

forecast = m.predict(future)

# Customization options

add_seasonality = st.checkbox("Add Seasonality")
if add_seasonality:
    yearly_seasonality = st.checkbox("Add Yearly Seasonality")
    if yearly_seasonality:
        m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
    else:
        m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=False)
else:
    m = Prophet()

trend = st.selectbox('Select trend direction', ['upward', 'downward', 'both'])
if trend == 'downward':
    m.changepoint_prior_scale = 0.5
elif trend == 'upward':
    m.changepoint_prior_scale = 2.0

m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

# Bar Plot
st.subheader('Bar Plot of forecast')
df_bar = forecast.tail(period).set_index('ds')
fig_bar = px.bar(df_bar, x=df_bar.index, y='yhat')
st.plotly_chart(fig_bar)

# Histogram
st.subheader('Histogram of forecast')
df_hist = forecast[['yhat']].tail(period)
fig_hist = px.histogram(df_hist, nbins=20, title='Histogram of Forecast')
st.plotly_chart(fig_hist)
