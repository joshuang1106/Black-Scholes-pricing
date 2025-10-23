from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
import streamlit as st
import plotly.express as px
import pandas as pd
import yfinance as yf

st.set_page_config(
    layout="wide",
)

def pricer(stock_price, strike_price, volatility, time, risk_free_interest):
    d1=1/np.sqrt(time)/volatility*(np.log(stock_price/strike_price)+(risk_free_interest+volatility**2/2)*time)
    d2=d1-volatility*np.sqrt(time)
    price_call=norm.cdf(d1)*stock_price-norm.cdf(d2)*strike_price*np.exp(-risk_free_interest*time)
    price_put= price_call -stock_price+strike_price*np.exp(-risk_free_interest*time)
    return price_call, price_put

st.title("Black Scholes Price")
col1,col2=st.columns(2)
S=col1.number_input("Current Stock Price",value=50.0)
K=col2.number_input("Strike Price",min_value=0.001, value=50.0, format="%.2f")
S2=col1.number_input("Volatility",min_value=0.0,value=0.3, format="%.2f")
T=col2.number_input("Time(year)",min_value=0.01,value=1.0, format="%.2f")
R=col1.number_input("risk free interest",min_value=0.0, value=0.04,format="%.2f")
call_price, put_price = pricer(S, K,S2, T, R)

st.write("### Option Pricing")
col1,col2=st.columns(2)
col1.metric(label="Call Option", value=f"${call_price:,.2f}")
col2.metric(label="Put Option", value=f"${put_price:,.2f}")

min_s, max_s = S-20,S+20
s_steps = 50

min_v, max_v = S2-0.1,S2+0.1
v_steps = 50

S_vals = np.linspace(min_s, max_s, s_steps)
V_vals = np.linspace(min_v, max_v, v_steps)

S, V = np.meshgrid(S_vals, V_vals)

Z_call_price = np.zeros_like(S)
Z_put_price = np.zeros_like(S)

for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        Z_call_price[i, j] = pricer(S[i, j], K, V[i, j], T, R)[0]
        Z_put_price[i,j] = pricer(S[i, j], K, V[i, j], T, R)[1]

st.subheader("3D Surface Plot: Call Option Price vs. Stock Price & Volatility")
fig_3d = go.Figure(data=[go.Surface(z=Z_call_price, x=S, y=V, colorscale='Viridis')])

fig_3d.update_layout(
    title='Call Option Price Surface',
    scene=dict(
        xaxis_title='Current Stock Price (S)',
        yaxis_title='Volatility (σ)',
        zaxis_title='Call Option Price',
        xaxis=dict(tickformat=".2f"),
        yaxis=dict(tickformat=".2f"),
        zaxis=dict(tickformat=".2f")
    ),
    autosize=True,
    height=600, # Adjusted height for side-by-side display
    margin=dict(l=65, r=50, b=65, t=90)
)

fig_3d2 = go.Figure(data=[go.Surface(z=Z_put_price, x=S, y=V, colorscale='Viridis')])

fig_3d2.update_layout(
    title='Put Option Price Surface',
    scene=dict(
        xaxis_title='Current Stock Price (S)',
        yaxis_title='Volatility (σ)',
        zaxis_title='Put Option Price',
        xaxis=dict(tickformat=".2f"),
        yaxis=dict(tickformat=".2f"),
        zaxis=dict(tickformat=".2f")
    ),
    autosize=True,
    height=600, 
    margin=dict(l=65, r=50, b=65, t=90)
)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_3d, use_container_width=True)
with col2:
    st.plotly_chart(fig_3d2, use_container_width=True)


stock = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", value="AAPL") 

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

if st.button("Download Stock Data"):
    if stock:
        try:
            data = yf.download(stock, start=start_date, end=end_date,auto_adjust=False)

            if not data.empty:
                data["percentage change"]=data['Adj Close'].pct_change()
                sigma=data["percentage change"].std()*np.sqrt(252)
                average_return=(1+data["percentage change"].mean())**252-1
                Sharpe_ratio= (average_return-0.0434)/sigma 
                st.line_chart(data['Adj Close'])
                startdata = float(data['Adj Close'].iloc[0])
                pricecall,priceput=pricer(startdata,startdata,sigma,1,0.0434)
                col1,col2=st.columns(2)
                col1.metric(label="Volatility(1 year)", value=f"{sigma:.2f}")
                col2.metric(label="Sharpe Ratio",value=f"{Sharpe_ratio:.2f}")
                
                st.write("### Black Scholes Option Pricing (1 year)")
                col1,col2=st.columns(2)
                col1.metric(label="Call Option", value=f"${float(pricecall):,.2f}")
                col2.metric(label="Put Option", value=f"${float(priceput):,.2f}")
                callprofit=float(data['Adj Close'].iloc[-1]) - float(data['Adj Close'].iloc[0])-float(pricecall)
                st.write("### Profit per share")
                col1,col2=st.columns(2)
                if float(data['Adj Close'].iloc[-1]) - float(data['Adj Close'].iloc[0])>=0:
                    col1.metric(label="Profit(call)" ,value=f"${callprofit:,.2f}")
                else:
                    col1.metric(label="Profit(call)" ,value=f"${-float(pricecall):,.2f}")
                
                Putprofit=-(float(data['Adj Close'].iloc[-1]) - float(data['Adj Close'].iloc[0]))-float(pricecall)

                if float(data['Adj Close'].iloc[0])-float(data['Adj Close'].iloc[-1]) >=0:
                    col2.metric(label="Profit(Put)" ,value=f"${Putprofit:,.2f}")
                else:
                    col2.metric(label="Profit(Put)" ,value=f"${-float(priceput):,.2f}")
            else:
                st.warning(f"No data found for ticker: {stock}. Please check the ticker symbol.")

        except Exception as e:
            st.error(f"An error occurred while downloading data: {e}")
            st.info("Please ensure the ticker symbol is correct.")
    else:
        st.warning("Please enter a stock ticker to download data.")