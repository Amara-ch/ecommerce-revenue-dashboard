import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import base64
from PIL import Image
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------- CONFIGURATION -------------------
st.set_page_config(page_title="E-commerce Revenue Prediction", layout="wide")

def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .main, .block-container {{
            background-color: rgba(13, 17, 23, 0.85);
            color: white;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("pic.jpg")

# ------------------- LOAD MODEL AND DATA -------------------
model = joblib.load("revenue_model.pkl")
features = joblib.load("model_features.pkl")

df = pd.read_csv("data.csv", encoding='ISO-8859-1')
df.dropna(subset=['CustomerID'], inplace=True)
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.month
df['Hour'] = df['InvoiceDate'].dt.hour
df['Weekday'] = df['InvoiceDate'].dt.weekday
df['Revenue'] = df['Quantity'] * df['UnitPrice']
df = df[(df['Revenue'] > 0) & (df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

country_cols = [col for col in features if col.startswith("Country_")]
country_names = [col.replace("Country_", "") for col in country_cols]

# ------------------- SIDEBAR NAVIGATION -------------------
# ------------------- SUPERCHARGED SIDEBAR NAVIGATION -------------------
with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-title {
        font-size:26px;
        font-weight:bold;
        color:#4CAF50;
        text-align:center;
        margin-bottom: 15px;
    }
    .sidebar-footer {
        margin-top: 30px;
        font-size: 13px;
        color: gray;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">E-Commerce Dashboard</div>', unsafe_allow_html=True)

    section = st.radio(" SELECT ONE :", [
        "Introduction",
        "EDA",
        "Model - Simple Prediction",
        "Model - Prediction with Gauge",
        "Statistical Analysis",
        "Conclusion"
    ])

    st.markdown('<div class="sidebar-footer">Project by Amara Tariq<br>Introduction to Data Science</div>', unsafe_allow_html=True)

# ------------------- INTRODUCTION -------------------
if section == "Introduction":
    st.title("📌 Project Overview")
    st.markdown("""
    Welcome to the E-commerce Revenue Prediction Dashboard!  
    This interactive app demonstrates how data science techniques can be applied to real-world e-commerce data to extract insights and build predictive models.

    ### 🔍 Objectives:
    - Perform exploratory data analysis (EDA) on retail transaction data
    - Use machine learning to predict revenue from transaction details
    - Visualize predictions interactively with dynamic gauges and charts

    ### 📁 Dataset Description:
    - Source: Kaggle - Online Retail Dataset  
      🔗 [Click here to view on Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
    - Records: Over 500,000 transactions from a UK-based online retailer
    - Features: Quantity, Unit Price, Country, Invoice Date, Customer ID, etc.

    ### 🧰 Tools & Technologies:
    - Streamlit for dashboard development
    - Pandas/Matplotlib/Seaborn for EDA
    - Scikit-learn for ML modeling
    - Plotly for interactive gauge visualization

    ### 📚 Academic Context:
    - Course: Introduction to Data Science  
    - Instructor: Dr. Nadeem Majeed  
    - Student: Amara Tariq

    ### 🎯 Value Addition:
    - Demonstrates real-world use of data science in business
    - Builds intuition on customer behavior and revenue drivers
    - Teaches dashboarding skills and app deployment
    - Encourages analytical thinking and problem-solving

    Dive in and explore how predictive analytics can transform business decision-making!
    """)


# ------------------- EDA -------------------
elif section == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Top Countries by Revenue")
    country_rev = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(country_rev)

    st.subheader("Revenue Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Revenue'], bins=100, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Revenue by Month")
    st.line_chart(df.groupby("Month")["Revenue"].sum())

    st.subheader("Revenue by Hour")
    st.line_chart(df.groupby("Hour")["Revenue"].sum())

    st.subheader("Revenue by Weekday")
    st.line_chart(df.groupby("Weekday")["Revenue"].sum())

    st.subheader("Word Cloud of Products")
    text = " ".join(df['Description'].dropna().astype(str))
    wc = WordCloud(width=800, height=400, background_color='black').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("Top 10 Selling Products")
    top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_products)

    st.subheader("Average Revenue by Country")
    avg_rev_country = df.groupby("Country")["Revenue"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(avg_rev_country)

    st.subheader("Heatmap: Hour vs Weekday Revenue")
    pivot = df.pivot_table(index='Hour', columns='Weekday', values='Revenue', aggfunc='sum').fillna(0)
    fig, ax = plt.subplots()
    sns.heatmap(pivot, cmap='YlGnBu', ax=ax)
    st.pyplot(fig)

    st.subheader("Monthly Transaction Count")
    st.line_chart(df.groupby("Month")["InvoiceNo"].count())

# ------------------- SIMPLE PREDICTION -------------------
elif section == "Model - Simple Prediction":
    st.title("⚙ Revenue Prediction")

    quantity = st.slider("Quantity", 1, 100, 1)
    unit_price = st.slider("Unit Price", 0.0, 100.0, 10.0)
    month = st.selectbox("Month", list(range(1, 13)))
    hour = st.selectbox("Hour", list(range(0, 24)))
    weekday = st.selectbox("Weekday", list(range(7)))
    country_selection = st.selectbox("Country", country_names)

    user_input = [quantity, unit_price, month, hour, weekday]
    for col in country_cols:
        user_input.append(1 if col == f"Country_{country_selection}" else 0)

    if st.button("Predict"):
        prediction = model.predict([user_input])[0]
        st.success(f"💰 Predicted Revenue: £{prediction:.2f}")

# ------------------- GAUGE PREDICTION -------------------
elif section == "Model - Prediction with Gauge":
    st.title("📊 Revenue Prediction with Gauge")

    quantity = st.slider("Quantity", 1, 100, 1, key="q")
    unit_price = st.slider("Unit Price", 0.0, 100.0, 10.0, key="u")
    month = st.selectbox("Month", list(range(1, 13)), key="m")
    hour = st.selectbox("Hour", list(range(0, 24)), key="h")
    weekday = st.selectbox("Weekday", list(range(7)), key="w")
    country_selection = st.selectbox("Country", country_names, key="c")

    user_input = [quantity, unit_price, month, hour, weekday]
    for col in country_cols:
        user_input.append(1 if col == f"Country_{country_selection}" else 0)

    if st.button("Predict with Gauge"):
        prediction = model.predict([user_input])[0]
        st.success(f"💰 Predicted Revenue: £{prediction:.2f}")

        color = "lime" if prediction < 100 else ("orange" if prediction < 500 else "red")
        gauge_width = min(628, prediction * 10)

        st.markdown(f"""
        <div style='text-align: center;'>
            <svg width="300" height="160">
              <circle cx="150" cy="150" r="100" fill="none" stroke="#ddd" stroke-width="30" />
              <circle cx="150" cy="150" r="100" fill="none" stroke="{color}" stroke-width="30"
                      stroke-dasharray="{gauge_width}, 999" transform="rotate(-90 150 150)" />
              <text x="150" y="130" font-size="20" fill="white" text-anchor="middle">£{prediction:.2f}</text>
            </svg>
        </div>
        """, unsafe_allow_html=True)

# ------------------- STATISTICAL ANALYSIS -------------------
elif section == "Statistical Analysis":
    st.title("📈 Statistical Analysis")

    # Encode country to match model features
    df_encoded = pd.get_dummies(df, columns=["Country"])

    # Add missing columns for alignment
    for col in features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Features and target
    X = df_encoded[features]
    y = df_encoded['Revenue']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Prediction
    y_pred = model.predict(X_test)

    # Error Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    percentage_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - percentage_error

    # Sizes
    total_records = len(df)
    train_count = len(X_train)
    test_count = len(X_test)
    train_percent = (train_count / total_records) * 100
    test_percent = (test_count / total_records) * 100

    st.markdown("### 📊 Data Split Summary")
    st.write(f"*Total Records:* {total_records}")
    st.write(f"*Training Set:* {train_count} records ({train_percent:.2f}%)")
    st.write(f"*Testing Set:* {test_count} records ({test_percent:.2f}%)")
    st.write(f"*Number of Features Used:* {X.shape[1]}")

    st.markdown("### 🧪 Model Evaluation Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📉 MAE", f"£{mae:.2f}")
        st.metric("📉 MSE", f"£{mse:.2f}")
        st.metric("📉 RMSE", f"£{rmse:.2f}")
    with col2:
        st.metric("🎯 R² Score", f"{r2:.4f}")
        st.metric("✅ Accuracy (Approx.)", f"{accuracy:.2f}%")
        st.metric("⚠ Avg. % Error", f"{percentage_error:.2f}%")

    st.markdown("### 📈 Actual vs Predicted Revenue")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6, color='lightblue', edgecolors='black')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=1)
    ax.set_xlabel("Actual Revenue")
    ax.set_ylabel("Predicted Revenue")
    ax.set_title("Predicted vs Actual Revenue")
    st.pyplot(fig)

    st.markdown("### 💡 Revenue Distribution Insight")
    st.write("Here's a quick look at the distribution of revenue values in the dataset:")
    st.write(df['Revenue'].describe())

    st.markdown("### ✅ Final Notes")
    st.success("Statistical analysis completed with full insights for evaluation.")
    st.markdown("### 🧮 Revenue Histogram")
    fig2, ax2 = plt.subplots()
    ax2.hist(df['Revenue'], bins=30, color='purple', edgecolor='white')
    ax2.set_title("Revenue Distribution")
    ax2.set_xlabel("Revenue Amount (£)")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

# ------------------- CONCLUSION -------------------
elif section == "Conclusion":
    st.title("✅ Conclusion")
    st.markdown("""
    This project demonstrates the power of data-driven decision making in the e-commerce industry. By combining exploratory analysis with machine learning, we have built an end-to-end pipeline to predict transaction revenue and visualize important trends.

    ### ✨ Key Learnings:
    - Cleaned and preprocessed messy transactional data
    - Explored patterns in sales using visual analytics
    - Built a predictive revenue model using supervised learning
    - Integrated model into an interactive Streamlit app with real-time input and feedback

    ### 📈 Business Impact:
    - Helps identify high-revenue transactions ahead of time
    - Supports inventory, marketing, and pricing strategies
    - Encourages data-driven culture in retail
    - Facilitates understanding of seasonal and country-wise trends

    ### 🚀 Future Work:
    - Deploy the model in production with live data
    - Integrate customer segmentation and RFM analysis
    - Explore deep learning for more complex purchase patterns
    - Expand dashboard with A/B testing results and promotion forecasts
    - Enhance UI/UX with more personalization and real-time alerts

    ### 🏁 Final Thoughts:
    This dashboard serves as a strong foundation for any e-commerce data science pipeline. With continued refinement and real-time integration, such tools can revolutionize strategic decision-making.

    🎯 Thank you for exploring this dashboard! Your feedback is welcome.
    """)