import numpy as np
import yfinance as yf
import pandas as pd
import plotly.express as px
import scipy.optimize
import streamlit as st
import plotly.graph_objects as go

# number of trading days in a year
NUM_TRADING_DAYS = 252
# we will generate random portfolios
NUM_PORTFOLIOS = 10000

stocks = ['AAPL', 'NVDA', 'MSFT', 'WMT', 'XOM', 'TSLA']
start_date = '2015-01-01'
end_date = '2023-01-01'

def download_data():
    stock_data = {}

    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

    return pd.DataFrame(stock_data)

def calculate_return(data):
    # calculate the logarithmic daily return 
    log_return = np.log(data/data.shift(1))
    # remove invalid values from first row
    return log_return[1:]

def generate_portfolios(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.dot(w.T, np.dot(returns.cov()*NUM_TRADING_DAYS, w)))

    return np.array(portfolio_means), np.array(portfolio_risks), np.array(portfolio_weights)

def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.dot(weights.T, np.dot(returns.cov()*NUM_TRADING_DAYS, weights))
    return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])

def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]

def optimize_portfolio(weights, returns):
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return scipy.optimize.minimize(fun=min_function_sharpe, x0=weights[0], args=returns,
                                   method='SLSQP', bounds=bounds, constraints=constraints)

def portfolio_optimization():
    # Download data
    data = download_data()
    returns = calculate_return(data)

    # Generate random portfolios
    means, risks, weights = generate_portfolios(returns)

    # Optimize portfolio
    optimal = optimize_portfolio(weights, returns)
    optimal_weights = optimal['x']

    # Create a plot for visualization
    create_plotly_visualization(means, risks, optimal_weights, returns)

    # Display the optimal portfolio statistics
    optimal_return, optimal_volatility, sharpe_ratio = statistics(optimal_weights, returns)
    st.write(f"Optimal Portfolio Expected Return: {optimal_return:.4f}")
    st.write(f"Optimal Portfolio Volatility: {optimal_volatility:.4f}")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.4f}")


def create_plotly_visualization(means, risks, optimal_weights, returns):
    # Create a scatter plot of the portfolios
    fig = go.Figure()

    # Add scatter trace for the portfolios
    fig.add_trace(go.Scatter(
        x=risks, y=means,
        mode='markers',
        marker=dict(color=means/risks, colorscale='Viridis', size=5),
        name='Portfolios'
    ))

    # Calculate optimal portfolio performance
    optimal_return, optimal_volatility, _ = statistics(optimal_weights, returns)

    # Add scatter trace for the optimal portfolio
    fig.add_trace(go.Scatter(
        x=[optimal_volatility], y=[optimal_return],
        mode='markers',
        marker=dict(color='green', size=10),
        name='Optimal Portfolio'
    ))

    # Update layout
    fig.update_layout(
        title='Portfolio Optimization using the Markowitz Model',
        xaxis_title='Volatility',
        yaxis_title='Return',
        coloraxis_colorbar=dict(title='Sharpe Ratio'),
        showlegend=True
    )

    return fig

def portfolio_optimization():
    # Download data
    data = download_data()
    returns = calculate_return(data)

    # Generate random portfolios
    means, risks, weights = generate_portfolios(returns)

    # Optimize portfolio
    optimal = optimize_portfolio(weights, returns)
    optimal_weights = optimal['x']
    
    display = ''
    for i in range(len(optimal_weights)):
        display += f'{stocks[i]}: {optimal_weights[i]} | '

    # Create a Plotly plot for visualization
    fig = create_plotly_visualization(means, risks, optimal_weights, returns)

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig)

    # Display the optimal portfolio statistics
    optimal_return, optimal_volatility, sharpe_ratio = statistics(optimal_weights, returns)
    st.write("This simulation works by randomly generating weights adding up to 1, and calculating the resulting expected return and volatility.")
    st.write("These portfolios are judged using the [Sharpe ratio](https://www.investopedia.com/terms/s/sharperatio.asp), which is the expected return divided by the volatility. A high sharpe ratio indicates that the portofilio has large returns given its level of risk")
    st.markdown(f"This page uses SciPy's optimization function to find the maximum Sharpe ratio **({sharpe_ratio:.4f})**. The portfolio with this ration has an expected return of {optimal_return:.4f} and volatility of {optimal_volatility:.4f}")

    fig2 = px.pie(
        names=[yf.Ticker(t).info["longName"] for t in stocks],
        values=optimal_weights,
        title="Optimal portfolio weights",
        hole=0  # Set this to 0 for a full pie chart; >0 for a donut chart
    )
    st.plotly_chart(fig2)


deflink = 'https://www.investopedia.com/terms/m/montecarlosimulation.asp#:~:text=A%20Monte%20Carlo%20simulation%20is%20a%20model%20used%20to%20predict%20the%20probability%20of%20a%20variety%20of%20outcomes%20when%20the%20potential%20for%20random%20variables%20is%20present.'

# Streamlit page configuration
st.title("Modern Portfolio Theory")
st.markdown("""The modern portfolio theory (MPT) is a practical method for selecting investments in order to maximize their overall returns while minimizing risk. This is achieved by building a diversified portfolio of financial assets.""")
st.markdown("This page will run a [Monte Carlo simulation](https://www.investopedia.com/terms/m/montecarlosimulation.asp#:~:text=A%20Monte%20Carlo%20simulation%20is%20a%20model%20used%20to%20predict%20the%20probability%20of%20a%20variety%20of%20outcomes%20when%20the%20potential%20for%20random%20variables%20is%20present.) to find a stock portfolio that balances risk and return.")
st.markdown("The return of a stock is measured as the natural logarithm of the closing price divided by yesterday's closing price for each trading day")
st.markdown("The risk (or volatility) of a portfolio is based on the variance of the daily return of its stocks and their weights")

stocks = st.text_input('Enter Stock Tickers (separated by commas)', 'AAPL,WMT,JNJ,BAC').split(',')

start_date = st.date_input('Start Date', value=pd.to_datetime('2013-01-01'))
end_date = st.date_input('End Date', value=pd.to_datetime('2024-01-01'))

NUM_PORTFOLIOS = st.slider("Portfolio count", 1, 10000, 1000)

if st.button('Run Simulation'):
    # Call the function to run the optimization and display results
    portfolio_optimization()
