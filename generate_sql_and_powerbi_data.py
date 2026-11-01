"""
generate_sql_and_powerbi_data.py
Populates:
1. sql/insert_data.sql (Sample batches of valid SQL INSERT statements for MySQL)
2. data/powerbi/Fact_Shipments.csv
3. data/powerbi/Dim_Route.csv
4. data/powerbi/Dim_ShippingMode.csv
5. data/powerbi/Dim_Date.csv
"""

import os
import joblib
import pandas as pd
import numpy as np

def generate_assets():
    df = pd.read_csv("data/processed/cleaned_delivery_data.csv")
    
    # Load trained model pipeline
    lr_pipeline = joblib.load("models/logistic_regression.pkl")
    
    # Feature columns used for prediction
    leakage_and_id_cols = [
        'Shipment_ID', 'Order_Date', 'Dispatch_Date', 'Delivery_Date',
        'Delivery_Status', 'Delivery_Delay_Days', 'Actual_Transit_Days',
        'Order_Year_Month', 'Delivery_Delayed'
    ]
    feature_cols = [col for col in df.columns if col not in leakage_and_id_cols]
    
    # Generate model predictions for the entire dataset
    probs = lr_pipeline.predict_proba(df[feature_cols])[:, 1]
    preds = lr_pipeline.predict(df[feature_cols])
    
    df['Delay_Probability'] = np.round(probs, 4)
    df['Predicted_Status'] = ['Delayed' if p == 1 else 'On Time' for p in preds]
    df['Predicted_Delayed'] = preds
    df['Prediction_Correct'] = (df['Predicted_Delayed'] == df['Delivery_Delayed']).astype(int)
    
    # Save enriched Fact Table for Power BI
    fact_path = "data/powerbi/Fact_Shipments.csv"
    df.to_csv(fact_path, index=False)
    print(f"Saved Fact_Shipments to {fact_path} ({len(df)} rows)")
    
    # 1. Dim_Route
    dim_route = df[['Origin', 'Destination', 'Distance', 'Distance_Tier']].drop_duplicates().reset_index(drop=True)
    dim_route['Route_Key'] = dim_route['Origin'] + " -> " + dim_route['Destination']
    dim_route.to_csv("data/powerbi/Dim_Route.csv", index=False)
    print(f"Saved Dim_Route ({len(dim_route)} routes)")
    
    # 2. Dim_ShippingMode
    shipping_modes_data = [
        {"Shipping_Mode": "Standard Ground", "SLA_Transit_Days": 5, "Cost_Tier": "Low", "Reliability_Index": 0.72},
        {"Shipping_Mode": "Express Air", "SLA_Transit_Days": 2, "Cost_Tier": "High", "Reliability_Index": 0.88},
        {"Shipping_Mode": "Same-Day Courier", "SLA_Transit_Days": 1, "Cost_Tier": "Premium", "Reliability_Index": 0.94},
        {"Shipping_Mode": "Freight Rail", "SLA_Transit_Days": 7, "Cost_Tier": "Economy", "Reliability_Index": 0.65}
    ]
    dim_mode = pd.DataFrame(shipping_modes_data)
    dim_mode.to_csv("data/powerbi/Dim_ShippingMode.csv", index=False)
    print("Saved Dim_ShippingMode")
    
    # 3. Dim_Date (Calendar Table for Power BI)
    min_date = pd.to_datetime(df['Order_Date']).min()
    max_date = pd.to_datetime(df['Delivery_Date']).max()
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    dim_date = pd.DataFrame({
        'Date': date_range.strftime('%Y-%m-%d'),
        'Year': date_range.year,
        'Quarter': 'Q' + date_range.quarter.astype(str),
        'Year_Quarter': date_range.year.astype(str) + '-Q' + date_range.quarter.astype(str),
        'Month': date_range.month,
        'Month_Name': date_range.strftime('%B'),
        'Year_Month': date_range.strftime('%Y-%m'),
        'Day': date_range.day,
        'Day_Of_Week': date_range.strftime('%A'),
        'Day_Number_Of_Week': date_range.dayofweek + 1,
        'Is_Weekend': (date_range.dayofweek >= 5).astype(int)
    })
    dim_date.to_csv("data/powerbi/Dim_Date.csv", index=False)
    print(f"Saved Dim_Date ({len(dim_date)} days)")
    
    # 4. Generate SQL Insert Script (sql/insert_data.sql)
    # Generate clean batch INSERT statements for MySQL (first 500 rows for instant testing + bulk guide)
    sql_path = "sql/insert_data.sql"
    with open(sql_path, "w", encoding="utf-8") as f:
        f.write("-- ==============================================================================\n")
        f.write("-- DATA INSERTION SCRIPT: supply_chain_analytics\n")
        f.write("-- Inserts verified shipment, transit, delivery, and ML inference records\n")
        f.write("-- ==============================================================================\n\n")
        f.write("USE supply_chain_analytics;\n\n")
        
        # Batch insert into Shipments
        f.write("-- 1. INSERT INTO Shipments\n")
        f.write("INSERT INTO Shipments (Shipment_ID, Order_Date, Dispatch_Date, Origin, Destination, Distance, Shipping_Mode, Dispatch_Latency_Days, Distance_Tier, Is_Weekend_Order) VALUES\n")
        shipment_rows = []
        for _, r in df.head(500).iterrows():
            shipment_rows.append(f"('{r['Shipment_ID']}', '{r['Order_Date']}', '{r['Dispatch_Date']}', '{r['Origin']}', '{r['Destination']}', {r['Distance']}, '{r['Shipping_Mode']}', {r['Dispatch_Latency_Days']}, '{r['Distance_Tier']}', {r['Is_Weekend_Order']})")
        f.write(",\n".join(shipment_rows) + ";\n\n")
        
        # Batch insert into Transportation
        f.write("-- 2. INSERT INTO Transportation\n")
        f.write("INSERT INTO Transportation (Shipment_ID, Shipping_Mode, Weather_Condition, Traffic_Level, Waiting_Time, Transportation_Cost, Cost_Per_KM, High_Waiting_Flag) VALUES\n")
        transport_rows = []
        for _, r in df.head(500).iterrows():
            transport_rows.append(f"('{r['Shipment_ID']}', '{r['Shipping_Mode']}', '{r['Weather_Condition']}', '{r['Traffic_Level']}', {r['Waiting_Time']}, {r['Transportation_Cost']}, {r['Cost_Per_KM']}, {r['High_Waiting_Flag']})")
        f.write(",\n".join(transport_rows) + ";\n\n")
        
        # Batch insert into Delivery_Records
        f.write("-- 3. INSERT INTO Delivery_Records\n")
        f.write("INSERT INTO Delivery_Records (Shipment_ID, Delivery_Date, Actual_Transit_Days, Delivery_Status, Delivery_Delay_Days, Delivery_Delayed) VALUES\n")
        delivery_rows = []
        for _, r in df.head(500).iterrows():
            delivery_rows.append(f"('{r['Shipment_ID']}', '{r['Delivery_Date']}', {r['Actual_Transit_Days']}, '{r['Delivery_Status']}', {r['Delivery_Delay_Days']}, {r['Delivery_Delayed']})")
        f.write(",\n".join(delivery_rows) + ";\n\n")
        
        # Batch insert into Prediction_Results
        f.write("-- 4. INSERT INTO Prediction_Results\n")
        f.write("INSERT INTO Prediction_Results (Shipment_ID, Predicted_Status, Delay_Probability, Model_Used, Prediction_Correct) VALUES\n")
        pred_rows = []
        for _, r in df.head(500).iterrows():
            pred_rows.append(f"('{r['Shipment_ID']}', '{r['Predicted_Status']}', {r['Delay_Probability']}, 'Logistic Regression', {r['Prediction_Correct']})")
        f.write(",\n".join(pred_rows) + ";\n\n")
        
        f.write("-- Note: To ingest all 6,500 records into MySQL in bulk, use the LOAD DATA INFILE command:\n")
        f.write("-- LOAD DATA INFILE 'data/powerbi/Fact_Shipments.csv' INTO TABLE ...\n")
        
    print(f"Generated sample SQL insertion script at {sql_path}")

if __name__ == "__main__":
    generate_assets()

