import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
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

def create_visualization(means, risks, optimal_weights, returns):
    plt.figure(figsize=(10, 6))
    plt.scatter(risks, means, c=means/risks, marker='o', label='Portfolios')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.colorbar(label='Sharpe Ratio')

    # Plot the optimal portfolio
    optimal_return, optimal_volatility, _ = statistics(optimal_weights, returns)
    plt.plot(optimal_volatility, optimal_return, 'g*', markersize=20.0, label='Optimal Portfolio')

    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

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
    create_visualization(means, risks, optimal_weights, returns)

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
        title='Portfolio Optimization with Monte Carlo Simulations',
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
    st.write(f"Optimal Portfolio Expected Return: {optimal_return:.4f}")
    st.write(f"Optimal Portfolio Volatility: {optimal_volatility:.4f}")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    st.write(f"Optimal weights: {display}")

# Streamlit page configuration
st.title("Portfolio Optimization with Monte Carlo Simulations")
st.markdown("""
This app uses Monte Carlo simulations to generate random portfolios from a selection of stocks. 
It then calculates the optimal portfolio based on the Sharpe ratio.
""")

# Call the function to run the optimization and display results
portfolio_optimization()