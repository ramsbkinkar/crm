from flask import Flask, render_template, jsonify, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import plotly.io as pio

# Create the Flask app instance
app = Flask(__name__)

# Load and clean data from Excel
file_path = 'data/OnlineRetail.xlsx'
df = pd.read_excel(file_path)

# Data preprocessing
df.dropna(inplace=True)  # Drop rows with missing values
df = df[df['Quantity'] > 0]  # Remove negative quantities
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# Feature Engineering for Customer Segmentation
customer_data = df.groupby('CustomerID').agg({'TotalAmount': 'sum', 'InvoiceDate': 'max'})

# Remove 'InvoiceDate' for scaling
customer_data = customer_data[['TotalAmount']]

# Scale the data
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data)

# KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=0)
customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

# Churn Prediction Model
churn_data = df.groupby('CustomerID').agg({'InvoiceDate': 'max'})
churn_data['Churn'] = churn_data['InvoiceDate'].apply(lambda x: 1 if (pd.to_datetime('today') - x).days > 30 else 0)

# Ensure churn has two classes for training the model
if churn_data['Churn'].nunique() == 1:
    churn_data['Churn'] = churn_data['Churn'].apply(lambda x: 1 if x == 1 else 0)

X = customer_data[['TotalAmount']]
y = churn_data['Churn']

# Check if there is more than one class in y
if len(y.unique()) > 1:
    model = LogisticRegression()
    model.fit(X, y)
else:
    print("Not enough class variance for training the model.")


# Routes
@app.route("/")
def home():
    # Create Plotly Graph for Customer Segments
    fig_segment = px.scatter(customer_data, x='TotalAmount', y='Cluster', color='Cluster',
                             title="Customer Segments (Clustering)", labels={'TotalAmount': 'Total Purchase Amount'})
    graph_html_segment = pio.to_html(fig_segment, full_html=False)

    # Create Plotly Graph for Sales by Customer
    fig_sales = px.bar(df, x='CustomerID', y='TotalAmount', title="Sales by Customer",
                       labels={'TotalAmount': 'Total Sales'})
    graph_html_sales = pio.to_html(fig_sales, full_html=False)

    return render_template('index.html',
                           graph_html_segment=graph_html_segment,
                           graph_html_sales=graph_html_sales)


@app.route("/customer_segment")
def customer_segment():
    segments = customer_data.reset_index().to_dict(orient='records')
    return jsonify(segments)


@app.route("/predict_churn", methods=["POST"])
def predict_churn():
    customer_id = request.json['CustomerID']
    if len(y.unique()) > 1:
        prediction = model.predict([customer_data.loc[customer_id].values])
        return jsonify({'churn_prediction': prediction[0]})
    else:
        return jsonify({'churn_prediction': 'Data insufficient for prediction'})


@app.route("/sales_forecast")
def sales_forecast():
    fig = px.bar(df, x='CustomerID', y='TotalAmount')
    graph_html = fig.to_html(full_html=False)
    return graph_html


if __name__ == "__main__":
    app.run(debug=True)
