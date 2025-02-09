import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from PIL import Image

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Shopping Trend Analysis", layout="wide")

# --- Custom Styling ---
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(to right, #f8f9fa, #e9ecef);
        }
        .stTitle {
            font-size: 30px !important;
            font-weight: bold !important;
            text-align: center !important;
            color: #212529 !important;
        }
        .stMarkdown {
            font-size: 16px !important;
            color: #6C757D !important;
            text-align: center !important;
        }
        .section-header {
            background: linear-gradient(to right, #007bff, #6610f2);
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 6px;
            border-radius: 5px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load dataset ---
df = pd.read_csv("shopping_trends.csv")

# --- Encode categorical variables ---
categorical_columns = ["Gender", "Category", "Season", "Subscription Status", "Shipping Type", "Payment Method", "Location"]
encoder_dict = {}

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoder_dict[col] = encoder

# --- Dashboard Title ---
st.markdown("<h1 class='stTitle'>🛒 Shopping Trend Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='stMarkdown'>Explore shopping trends, spending habits, and future purchase predictions!</p>", unsafe_allow_html=True)

# --- View Raw Data ---
if st.checkbox("📜 Show Raw Data"):
    st.write(df.head())

# --- STEP 1: Customer Demographics & Preferences ---
st.markdown("<div class='section-header'>1️⃣ Customer Demographics & Preferences</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("👨‍👩‍👧‍👦 Gender Breakdown")
    gender_counts = df["Gender"].value_counts()
    fig_gender, ax = plt.subplots(figsize=(2, 1.5))
    ax.pie(
        gender_counts, labels=encoder_dict["Gender"].inverse_transform(gender_counts.index), autopct='%1.1f%%',
        colors=["#E63946", "#457B9D"], startangle=140, wedgeprops={'edgecolor': 'black'}, textprops={'fontsize': 6}
    )
    ax.axis("equal")
    st.pyplot(fig_gender)

with col2:
    st.subheader("⭐ Subscription Status")
    subscription_counts = df["Subscription Status"].value_counts()
    fig_subs, ax = plt.subplots(figsize=(2, 1.5))
    ax.pie(
        subscription_counts, labels=encoder_dict["Subscription Status"].inverse_transform(subscription_counts.index), autopct='%1.1f%%',
        colors=["#2A9D8F", "#E76F51"], startangle=140, wedgeprops={'edgecolor': 'black'}, textprops={'fontsize': 6}
    )
    ax.axis("equal")
    st.pyplot(fig_subs)

# --- STEP 2: Best-Selling Products & Popular Locations ---
st.markdown("<div class='section-header'>2️⃣ Best-Selling Products & Top Locations</div>", unsafe_allow_html=True)

st.subheader("🔥 Top 10 Most Purchased Items")
top_products = df["Item Purchased"].value_counts().head(10)
fig_products, ax = plt.subplots(figsize=(4, 2))
sns.barplot(x=top_products.index, y=top_products.values, hue=top_products.index, palette="viridis", legend=False, ax=ax)
plt.xticks(rotation=30, fontsize=8, ha="right")
plt.yticks(fontsize=8)
plt.xlabel("Item Purchased", fontsize=9)
plt.ylabel("Total Purchases", fontsize=9)
plt.tight_layout()
st.pyplot(fig_products)

# --- Top Locations ---
st.subheader("📍 Where People Spend the Most")
location_sales = df.groupby("Location")["Purchase Amount (USD)"].sum().sort_values(ascending=False).head(10)
fig_locations, ax = plt.subplots(figsize=(4, 2))
sns.barplot(x=encoder_dict["Location"].inverse_transform(location_sales.index), y=location_sales.values, hue=encoder_dict["Location"].inverse_transform(location_sales.index), palette="magma", legend=False, ax=ax)
plt.xticks(rotation=30, fontsize=8, ha="right")
plt.yticks(fontsize=8)
plt.xlabel("Locations", fontsize=9)
plt.ylabel("Total Sales (USD)", fontsize=9)
plt.tight_layout()
st.pyplot(fig_locations)

# --- STEP 3: Filter Data Before Download ---
st.markdown("<div class='section-header'>3️⃣ Filter & Download Data</div>", unsafe_allow_html=True)

selected_season = st.selectbox("Select Season:", encoder_dict["Season"].inverse_transform(df["Season"].unique()))
selected_payment = st.selectbox("Select Payment Method:", encoder_dict["Payment Method"].inverse_transform(df["Payment Method"].unique()))
selected_shipping = st.selectbox("Select Shipping Type:", encoder_dict["Shipping Type"].inverse_transform(df["Shipping Type"].unique()))

filtered_df = df[
    (df["Season"] == encoder_dict["Season"].transform([selected_season])[0]) &
    (df["Payment Method"] == encoder_dict["Payment Method"].transform([selected_payment])[0]) &
    (df["Shipping Type"] == encoder_dict["Shipping Type"].transform([selected_shipping])[0])
]

st.subheader("Filtered Data")
st.write(filtered_df)

filtered_csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(label="⬇️ Download Filtered Data", data=filtered_csv, file_name="filtered_shopping_data.csv", mime="text/csv")

# --- STEP 4: Download Visualizations ---
st.subheader("📷 Download Charts as Images")
for fig_name, fig in [
    ("Gender Breakdown", fig_gender),
    ("Subscription Status", fig_subs),
    ("Most Popular Products", fig_products),
    ("Highest Spending Locations", fig_locations)
]:
    fig.savefig(f"{fig_name}.jpg", format="jpg", dpi=300, bbox_inches="tight")
    with open(f"{fig_name}.jpg", "rb") as img:
        st.download_button(label=f"⬇️ {fig_name}", data=img, file_name=f"{fig_name}.jpg", mime="image/jpeg")