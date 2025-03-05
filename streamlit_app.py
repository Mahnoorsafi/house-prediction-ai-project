import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import statsmodels.api as sm

# Load the trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Title and description
st.title("🏡 House Price Prediction App")
st.write("Enter the house details below, and our model will predict the estimated price!")

# Sidebar for user inputs
st.sidebar.header("🏠 House Features")
area = st.sidebar.number_input("🏡 Area (sq ft)", min_value=500, max_value=10000, value=1500)
bedrooms = st.sidebar.slider("🛏 Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.sidebar.slider("🛁 Bathrooms", min_value=1, max_value=5, value=2)
stories = st.sidebar.slider("🏢 Stories", min_value=1, max_value=4, value=1)

# Categorical inputs (converted to binary)
mainroad = st.sidebar.selectbox("🚗 Connected to Main Road?", ["No", "Yes"])
guestroom = st.sidebar.selectbox("🛏 Has Guest Room?", ["No", "Yes"])
basement = st.sidebar.selectbox("🏠 Has Basement?", ["No", "Yes"])
hotwaterheating = st.sidebar.selectbox("🔥 Has Hot Water Heating?", ["No", "Yes"])
airconditioning = st.sidebar.selectbox("❄ Has Air Conditioning?", ["No", "Yes"])

# Convert categorical inputs to numerical values
mainroad = 1 if mainroad == "Yes" else 0
guestroom = 1 if guestroom == "Yes" else 0
basement = 1 if basement == "Yes" else 0
hotwaterheating = 1 if hotwaterheating == "Yes" else 0
airconditioning = 1 if airconditioning == "Yes" else 0

# Prediction function
def predict_price():
    input_data = np.array(
        [[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning]]
    )
    prediction = model.predict(input_data)[0]
    return round(prediction, 2)

# Prediction button
if st.sidebar.button("🔮 Predict Price"):
    estimated_price = predict_price()
    st.subheader(f"💰 Estimated House Price: ${estimated_price}")

# Sample dataset for visualization
data = {
    "Area": np.random.randint(500, 5000, 100),
    "Bedrooms": np.random.randint(1, 5, 100),
    "Bathrooms": np.random.randint(1, 4, 100),
    "Stories": np.random.randint(1, 4, 100),
    "Price": np.random.randint(50000, 500000, 100)
}

df = pd.DataFrame(data)

# Scatter plot with Linear Regression
st.subheader("📈 House Price Scatter Plot with Linear Regression")

# Choose X-axis variable dynamically
x_var = st.selectbox("Select a variable for X-axis:", ["Area", "Bedrooms", "Bathrooms", "Stories"])

# Add a constant for intercept in regression
X = sm.add_constant(df[x_var])  # Independent Variable + Constant
y = df["Price"]  # Dependent Variable

# Fit regression model
model_lr = sm.OLS(y, X).fit()
df["Trendline"] = model_lr.predict(X)

# Plot scatter with regression trendline
fig = px.scatter(df, x=x_var, y="Price", color="Price", size="Price",
                 title=f"House Price vs {x_var}",
                 labels={x_var: x_var, "Price": "Price"})

# Add regression line
fig.add_traces(px.line(df, x=x_var, y="Trendline").data)

# Show graph
st.plotly_chart(fig)

# Footer
st.markdown("---")
st.write("📌 *Powered by Machine Learning & Streamlit*")
