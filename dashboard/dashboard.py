
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from babel.numbers import format_currency
sns.set(style='dark')

def create_daily_orders_df(df):
    
    daily_orders_df = df.resample(rule='M', on='order_estimated_delivery_date').agg({
    "order_id": "nunique",
    "price": "sum"
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
    "order_id": "order_count",
    "price": "revenue"
    }, inplace=True)

    return daily_orders_df 

def create_bystate_df(df):
    bystate_df = df.groupby(by="customer_state").customer_id.nunique().reset_index()
    bystate_df.rename(columns={
        "customer_id": "customer_count"
    }, inplace=True)
    
    return bystate_df

def create_byorder_df(df):
    byorder_df = df.groupby(by="product_category_name").order_id.nunique().reset_index()
    byorder_df.rename(columns={
        "order_id": "order_count"
    }, inplace=True)
    byorder_df = byorder_df.sort_values(by="order_count", ascending=False)
    return byorder_df

def create_rfm_df(df):
    rfm_df = df.groupby(by="customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max",
        "order_id": "nunique",
        "price": "sum"
    })
    rfm_df.columns = ["customer_id", "order_purchase_timestamp", "frequency", "monetary"]
    
    rfm_df["order_purchase_timestamp"] = rfm_df["order_purchase_timestamp"].dt.date
    recent_date = df["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["order_purchase_timestamp"].apply(lambda x: (recent_date - x).days)
    rfm_df.drop("order_purchase_timestamp", axis=1, inplace=True)
    rfm_df.sort_values(by="recency", ascending=True).head(5)
    
    return rfm_df

### EDA
all_df = pd.read_csv("dashboard/ds_data.csv")

datetime_columns = ["order_purchase_timestamp", "order_estimated_delivery_date"]
all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(inplace=True)

for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

### Data Filter
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

with st.sidebar:

    st.title('SELECT DATE')
    
    ### Filter start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Time Range',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) & 
                (all_df["order_purchase_timestamp"] <= str(end_date))]

### st.dataframe(main_df)

### DataFrame Preparation
daily_orders_df = create_daily_orders_df(main_df)
bystate_df = create_bystate_df(main_df)
byorder_df = create_byorder_df(main_df)
rfm_df = create_rfm_df(main_df)

st.header('SALES PERFORMANCE DASHBOARD')

st.subheader('Order Performance')

col1, col2 = st.columns(2)

with col1:
    total_orders = daily_orders_df.order_count.sum()
    st.metric("Total Order in (2016 - 2018)", value=total_orders)

with col2:
    total_revenue = format_currency(daily_orders_df.revenue.sum(), "USD", locale='es_CO') 
    st.metric("Total Revenue in (2016 - 2018)", value=total_revenue)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_orders_df["order_estimated_delivery_date"],
    daily_orders_df["order_count"],
    marker='o', 
    linewidth=2,
    color="#355E3B"
)
ax.set_title(None)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
st.pyplot(fig)

### Customer Distribution
st.subheader("Customer Distribution by State")

with col1:
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = sns.color_palette("Set2")
    sns.barplot(
    y="customer_count", 
    x="customer_state",
    data=bystate_df.sort_values(by="customer_count", ascending=False),
    palette=colors,
    ax=ax
)
ax.set_title(None)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
st.pyplot(fig)

def generate_top_bottom_products(data): 
    byorder_df = data.groupby("product_category_name").order_id.count().reset_index()
    byorder_df.rename(columns={"order_id": "order_count"}, inplace=True)
    byorder_df = byorder_df.sort_values(by="order_count", ascending=False)
    top_10_products = byorder_df.head(10)
    bottom_10_products = byorder_df.tail(10)
    return top_10_products, bottom_10_products
data = pd.read_csv("dashboard/ds_data.csv")
top_10_products, bottom_10_products = generate_top_bottom_products(data)
st.subheader("Top 10 Products by Order")
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.barplot(
    x="order_count",
    y="product_category_name",
    data=top_10_products,
    palette="Blues_r"
)
plt.title(None)
plt.xlabel(None)
plt.ylabel(None)
st.pyplot(fig1)
st.subheader("Bottom 10 Products by Order")
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.barplot(
    x="order_count",
    y="product_category_name",
    data=bottom_10_products,
    palette="Reds_r"
)
plt.title(None)
plt.xlabel(None)
plt.ylabel(None)
st.pyplot(fig2)

### RFM Analysis
st.subheader("RFM Analysis")
col1, col2, col3 = st.columns(3)
with col1:
    avg_recency = round(rfm_df.recency.mean(), 1)
    st.metric("Average Recency", value=avg_recency)
with col2:
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    st.metric("Average Frequency", value=avg_frequency)
with col3:
    avg_frequency = format_currency(rfm_df.monetary.mean(), "USD", locale='es_CO') 
    st.metric("Average Monetary", value=avg_frequency)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(35, 15))
colors = ["#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9"]
sns.barplot(y="recency", x="customer_id", data=rfm_df.sort_values(by="recency", ascending=False).head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Customer ID", fontsize=30)
ax[0].set_title("By Recency", loc="center", fontsize=30)
ax[0].tick_params(axis='y', labelsize=30)
ax[0].tick_params(axis='x', labelsize=35, rotation=90)
sns.barplot(y="frequency", x="customer_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Customer ID", fontsize=30)
ax[1].set_title("By Frequency", loc="center", fontsize=30)
ax[1].tick_params(axis='y', labelsize=30)
ax[1].tick_params(axis='x', labelsize=35, rotation=90)
sns.barplot(y="monetary", x="customer_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), palette=colors, ax=ax[2])
ax[2].set_ylabel(None)
ax[2].set_xlabel("Customer ID", fontsize=30)
ax[2].set_title("By Monetary", loc="center", fontsize=30)
ax[2].tick_params(axis='y', labelsize=30)
ax[2].tick_params(axis='x', labelsize=35, rotation=90)
st.pyplot(fig)


### RFM Scores
st.subheader('Segmented based on RFM')

### RFM scoring
rfm_df['r_rank'] = rfm_df['recency'].rank(ascending=False)
rfm_df['f_rank'] = rfm_df['frequency'].rank(ascending=True)
rfm_df['m_rank'] = rfm_df['monetary'].rank(ascending=True)
### Normalization Ranked
rfm_df['r_rank_norm'] = (rfm_df['r_rank'] / rfm_df['r_rank'].max()) * 100
rfm_df['f_rank_norm'] = (rfm_df['f_rank'] / rfm_df['f_rank'].max()) * 100
rfm_df['m_rank_norm'] = (rfm_df['m_rank'] / rfm_df['m_rank'].max()) * 100
rfm_df.drop(columns=['r_rank', 'f_rank', 'm_rank'], inplace=True)
# Calculated RFM Score
rfm_df['RFM_score'] = 0.15 * rfm_df['r_rank_norm'] + 0.28 * rfm_df['f_rank_norm'] + 0.57 * rfm_df['m_rank_norm']
rfm_df['RFM_score'] *= 0.05
rfm_df = rfm_df.round(2)
# Segmented based on RFM
rfm_df["customer_segment"] = np.where(
    rfm_df['RFM_score'] > 4.5, "Top customers", (np.where(
        rfm_df['RFM_score'] > 4, "High-value customer", (np.where(
            rfm_df['RFM_score'] > 3, "Medium-value customer", np.where(
                rfm_df['RFM_score'] > 1.6, 'Low-value customers', 'Lost customers'))))))
### Segmented Bar chart
customer_segment_df = rfm_df.groupby(by="customer_segment", as_index=False).customer_id.nunique()
customer_segment_df['customer_segment'] = pd.Categorical(customer_segment_df['customer_segment'], [
    "Lost customers", "Low-value customers", "Medium-value customer",
    "High-value customer", "Top customers"
])
# Plot Segmented for Customer
plt.figure(figsize=(10, 5))
colors = sns.color_palette("Set2")
sns.barplot(
    x="customer_id",
    y="customer_segment",
    data=customer_segment_df.sort_values(by="customer_segment", ascending=False),
    palette=colors
)
plt.title(None)
plt.ylabel(None)
plt.xlabel(None)
plt.tick_params(axis='y', labelsize=12)
st.pyplot(plt)

st.caption('Copyright (c) justfeds 2023')
