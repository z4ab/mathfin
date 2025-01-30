import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

NUM_OF_SIMULATIONS = 100

def stock_monte_carlo(s0, mean, sigma, N=252):
    result = []

    for _ in range(NUM_OF_SIMULATIONS):
        prices = [s0]

        for _ in range(N):
            prev = prices[-1]
            s = prev * np.exp((mean - 0.5 * sigma ** 2) + sigma * np.random.normal())
            prices.append(s)
        result.append(prices)
    
    data = pd.DataFrame(result)
    data = data.T # transpose

    data['mean'] = data.mean(axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(N)), y=data['mean'],
        mode = 'lines'
    ))
    fig.update_layout(
        title='Stock Price Simulation using GBM Model',
        xaxis_title='Trading day',
        yaxis_title='Stock price',
    )
    st.plotly_chart(fig)

st.title("Geometric Brownian Motion")
st.markdown("The random walk theory suggests that stock prices follow a changes in prices are random, so past prices can not be used to determine future prices")
st.markdown("The geometric Browninan motion (GBM) is a model where stock prices follow a random walk, and the Efficient Market Hypothesis is assumed. This means that stock prices are assumed to always be traded at their market value, and there are no undervalued or overvalued stocks.")
st.markdown("In this page, we will run simulate a historical stock price graph based on the mean and standard deviation of the stock price")

initial_price = st.number_input("Initial price", value=100)
mean = st.number_input("Mean", format="%0.4f", value=0.0004)
stdev = st.number_input("Standard Deviation", format="%0.4f", value=0.02)

if st.button('Run Simulation'):
    stock_monte_carlo(initial_price, mean, stdev)   