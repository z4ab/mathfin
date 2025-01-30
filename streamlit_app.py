import streamlit as st
pages = [
    st.Page("./pages/MPT.py", title="Stock Portfolio Generation"),
    st.Page("./pages/stock-price.py", title="Stock Price Simulation")
]
pg = st.navigation(pages)
pg.run()