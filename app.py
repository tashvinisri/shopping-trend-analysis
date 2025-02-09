import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import io  # For handling image download

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="🛒 Shopping Trend Analysis", layout="wide")

# --- Page Title & Description ---
st.markdown("""
    <h1 style='text-align: center; color: #007bff;'>🛒 Shopping Trend Analysis</h1>
    <p style='text-align: center; font-size: 18px; color: #6C757D;'>
        Explore shopping trends, spending habits, and predict future purchases with advanced analytics.
    </p>
""", unsafe_allow_html=True)

# --- Load dataset ---
df = pd.read_csv("shopping_trends.csv")

# --- Encode categorical variables ---
categorical_columns = ["Gender", "Category", "Season", "Subscription Status", "Location"]
encoder_dict = {}

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    df[f"{col}_Name"] = encoder.inverse_transform(df[col])  # Store actual names
    encoder_dict[col] = encoder

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Data")
selected_category = st.sidebar.multiselect("Select Category", df["Category_Name"].unique())
selected_location = st.sidebar.multiselect("Select Location", df["Location_Name"].unique())

filtered_df = df.copy()
if selected_category:
    filtered_df = filtered_df[filtered_df["Category_Name"].isin(selected_category)]
if selected_location:
    filtered_df = filtered_df[filtered_df["Location_Name"].isin(selected_location)]

# --- Create Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Summary Metrics", "📈 Visualizations", "🔮 Predict Future Purchases"])

# --- STEP 1: Summary Metrics ---
with tab1:
    st.subheader("📊 Key Metrics")
    total_sales = filtered_df["Purchase Amount (USD)"].sum()
    total_transactions = filtered_df.shape[0]
    total_quantity_sold = filtered_df["Previous Purchases"].sum()
    
    st.metric(label="💰 Total Sales", value=f"${total_sales:,.2f}")
    st.metric(label="🛍 Total Transactions", value=f"{total_transactions}")
    st.metric(label="📦 Total Quantity Sold", value=f"{total_quantity_sold}")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👨‍👩‍👧‍👦 Gender Breakdown")
        gender_counts = filtered_df["Gender_Name"].value_counts()
        fig_gender, ax = plt.subplots()
        ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=["#E63946", "#457B9D"], startangle=140, wedgeprops={'edgecolor': 'black'})
        ax.axis("equal")
        st.pyplot(fig_gender)

        buf = io.BytesIO()
        fig_gender.savefig(buf, format="jpg")
        buf.seek(0)
        st.download_button(label="📥 Download Gender Chart", data=buf, file_name="gender_distribution.jpg", mime="image/jpeg")

    with col2:
        st.subheader("⭐ Subscription Status")
        subscription_counts = filtered_df["Subscription Status_Name"].value_counts()
        fig_subs, ax = plt.subplots()
        ax.pie(subscription_counts, labels=subscription_counts.index, autopct='%1.1f%%', colors=["#2A9D8F", "#E76F51"], startangle=140, wedgeprops={'edgecolor': 'black'})
        ax.axis("equal")
        st.pyplot(fig_subs)

        buf = io.BytesIO()
        fig_subs.savefig(buf, format="jpg")
        buf.seek(0)
        st.download_button(label="📥 Download Subscription Chart", data=buf, file_name="subscription_status.jpg", mime="image/jpeg")

    if st.checkbox("📜 Show Raw Data"):
        st.dataframe(filtered_df.head())

# --- STEP 2: Visualizations ---
with tab2:
    st.subheader("📈 Shopping Trends")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Top 10 Most Purchased Items")
        top_products = filtered_df["Item Purchased"].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(y=top_products.index, x=top_products.values, palette="viridis", ax=ax)
        ax.set_xlabel("Purchase Count")
        ax.set_ylabel("Item Purchased")
        ax.set_title("Top 10 Most Purchased Items")
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="jpg")
        buf.seek(0)
        st.download_button(label="📥 Download Top Items Chart", data=buf, file_name="top_purchased_items.jpg", mime="image/jpeg")

    with col2:
        st.subheader("📍 Top 10 Locations by Sales")
        top_locations = filtered_df.groupby("Location_Name")["Purchase Amount (USD)"].sum().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(y=top_locations.index, x=top_locations.values, palette="magma", ax=ax)
        ax.set_xlabel("Total Sales (USD)")
        ax.set_ylabel("Location")
        ax.set_title("Top 10 Locations by Sales")
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="jpg")
        buf.seek(0)
        st.download_button(label="📥 Download Sales Chart", data=buf, file_name="top_locations_sales.jpg", mime="image/jpeg")

# --- STEP 3: Predict Future Purchases ---
with tab3:
    st.subheader("🔮 Predict Future Purchase Amount")

    features = ["Age", "Previous Purchases", "Gender", "Category", "Season", "Subscription Status", "Location"]
    X = df[features]
    y = df["Purchase Amount (USD)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train_scaled, y_train)

    age = st.slider("Select Age:", 18, 70, 30)
    prev_purchases = st.slider("Previous Purchases:", 0, 50, 5)
    location = st.selectbox("Location:", df["Location_Name"].unique())
    gender = st.selectbox("Gender:", df["Gender_Name"].unique())
    category = st.selectbox("Category:", df["Category_Name"].unique())
    season = st.selectbox("Season:", df["Season_Name"].unique())
    subscription = st.selectbox("Subscription Status:", df["Subscription Status_Name"].unique())

    prediction = model.predict(scaler.transform(np.array([[age, prev_purchases, 
        encoder_dict['Gender'].transform([gender])[0], 
        encoder_dict['Category'].transform([category])[0], 
        encoder_dict['Season'].transform([season])[0], 
        encoder_dict['Subscription Status'].transform([subscription])[0], 
        encoder_dict['Location'].transform([location])[0]]])))[0]

    st.subheader(f"🛍 Predicted Purchase Amount: **${prediction:.2f}**")