-- ==============================================================================
-- ANALYTICAL SQL QUERIES: supply_chain_analytics
-- Comprehensive Business Intelligence, SLA Performance & Machine Learning Audit
-- ==============================================================================

USE supply_chain_analytics;

-- ------------------------------------------------------------------------------
-- QUERY 1: Executive KPI Summary (Total Volume, On-Time Rate & Delay Rate)
-- Business Value: Baseline executive scorecard metric for overall logistics SLA.
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(s.Shipment_ID) AS Total_Shipments,
    SUM(CASE WHEN d.Delivery_Delayed = 0 THEN 1 ELSE 0 END) AS On_Time_Shipments,
    SUM(CASE WHEN d.Delivery_Delayed = 1 THEN 1 ELSE 0 END) AS Delayed_Shipments,
    ROUND(AVG(CASE WHEN d.Delivery_Delayed = 0 THEN 1.0 ELSE 0.0 END) * 100, 2) AS On_Time_Delivery_Rate_Pct,
    ROUND(AVG(CASE WHEN d.Delivery_Delayed = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) AS Delay_Rate_Pct
FROM Shipments s
INNER JOIN Delivery_Records d ON s.Shipment_ID = d.Shipment_ID;


-- ------------------------------------------------------------------------------
-- QUERY 2: On-Time Delivery Percentage by Origin Hub (Regional Sourcing Performance)
-- Business Value: Identifies warehouse hubs with chronic dispatch or local freight bottlenecks.
-- ------------------------------------------------------------------------------
SELECT 
    s.Origin,
    COUNT(s.Shipment_ID) AS Total_Dispatched,
    SUM(CASE WHEN d.Delivery_Delayed = 0 THEN 1 ELSE 0 END) AS On_Time_Deliveries,
    SUM(CASE WHEN d.Delivery_Delayed = 1 THEN 1 ELSE 0 END) AS Delayed_Deliveries,
    ROUND(AVG(CASE WHEN d.Delivery_Delayed = 0 THEN 1.0 ELSE 0.0 END) * 100, 2) AS On_Time_Rate_Pct,
    ROUND(AVG(t.Waiting_Time), 2) AS Avg_Dock_Waiting_Hours
FROM Shipments s
INNER JOIN Delivery_Records d ON s.Shipment_ID = d.Shipment_ID
INNER JOIN Transportation t ON s.Shipment_ID = t.Shipment_ID
GROUP BY s.Origin
ORDER BY On_Time_Rate_Pct DESC;


-- ------------------------------------------------------------------------------
-- QUERY 3: Delayed Shipments by Destination City & Average Delay Severity
-- Business Value: Pinpoints destination markets requiring expedited carrier contracts.
-- ------------------------------------------------------------------------------
SELECT 
    s.Destination,
    COUNT(s.Shipment_ID) AS Total_Received,
    SUM(d.Delivery_Delayed) AS Total_Delayed,
    ROUND(AVG(d.Delivery_Delayed) * 100, 2) AS Delay_Rate_Pct,
    ROUND(AVG(CASE WHEN d.Delivery_Delayed = 1 THEN d.Delivery_Delay_Days ELSE NULL END), 2) AS Avg_Days_Late
FROM Shipments s
INNER JOIN Delivery_Records d ON s.Shipment_ID = d.Shipment_ID
GROUP BY s.Destination
ORDER BY Delay_Rate_Pct DESC;


-- ------------------------------------------------------------------------------
-- QUERY 4: Average Transit Duration and Cost Efficiency by Shipping Mode
-- Business Value: Benchmark SLA reliability versus financial cost across freight modalities.
-- ------------------------------------------------------------------------------
SELECT 
    s.Shipping_Mode,
    COUNT(s.Shipment_ID) AS Shipment_Count,
    ROUND(AVG(s.Distance), 1) AS Avg_Distance_KM,
    ROUND(AVG(d.Actual_Transit_Days), 2) AS Avg_Transit_Days,
    ROUND(AVG(t.Transportation_Cost), 2) AS Avg_Cost_USD,
    ROUND(AVG(t.Cost_Per_KM), 3) AS Avg_Cost_Per_KM,
    ROUND(AVG(d.Delivery_Delayed) * 100, 2) AS Delay_Rate_Pct
FROM Shipments s
INNER JOIN Transportation t ON s.Shipment_ID = t.Shipment_ID
INNER JOIN Delivery_Records d ON s.Shipment_ID = d.Shipment_ID
GROUP BY s.Shipping_Mode
ORDER BY Avg_Cost_USD DESC;


-- ------------------------------------------------------------------------------
-- QUERY 5: Transportation Cost Breakdown by Distance Tier & Traffic Congestion
-- Business Value: Shows how heavy traffic amplifies transportation cost surcharges across haul lengths.
-- ------------------------------------------------------------------------------
SELECT 
    s.Distance_Tier,
    t.Traffic_Level,
    COUNT(s.Shipment_ID) AS Volume,
    ROUND(AVG(t.Transportation_Cost), 2) AS Avg_Total_Cost,
    ROUND(AVG(t.Waiting_Time), 2) AS Avg_Dock_Wait_Hours,
    ROUND(AVG(d.Delivery_Delayed) * 100, 2) AS Delay_Rate_Pct
FROM Shipments s
INNER JOIN Transportation t ON s.Shipment_ID = t.Shipment_ID
INNER JOIN Delivery_Records d ON s.Shipment_ID = d.Shipment_ID
GROUP BY s.Distance_Tier, t.Traffic_Level
ORDER BY s.Distance_Tier, FIELD(t.Traffic_Level, 'Low', 'Medium', 'High', 'Severe');


-- ------------------------------------------------------------------------------
-- QUERY 6: Monthly Delivery Trend Analysis (Volume, Cost, and On-Time Stability)
-- Business Value: Evaluates seasonal trends, holiday spikes, and quarter-over-quarter SLA shifts.
-- ------------------------------------------------------------------------------
SELECT 
    DATE_FORMAT(s.Order_Date, '%Y-%m') AS Order_Year_Month,
    COUNT(s.Shipment_ID) AS Monthly_Volume,
    ROUND(SUM(t.Transportation_Cost), 2) AS Total_Freight_Spend,
    ROUND(AVG(t.Transportation_Cost), 2) AS Avg_Cost_Per_Shipment,
    ROUND(AVG(CASE WHEN d.Delivery_Delayed = 0 THEN 1.0 ELSE 0.0 END) * 100, 2) AS OTDR_Pct,
    ROUND(AVG(d.Delivery_Delay_Days), 2) AS Avg_Delay_Days
FROM Shipments s
INNER JOIN Transportation t ON s.Shipment_ID = t.Shipment_ID
INNER JOIN Delivery_Records d ON s.Shipment_ID = d.Shipment_ID
GROUP BY DATE_FORMAT(s.Order_Date, '%Y-%m')
ORDER BY Order_Year_Month ASC;


-- ------------------------------------------------------------------------------
-- QUERY 7: Weather-Wise Delay Impact and Average Waiting Time
-- Business Value: Quantifies environmental vulnerability to justify weather-contingency buffering.
-- ------------------------------------------------------------------------------
SELECT 
    t.Weather_Condition,
    COUNT(s.Shipment_ID) AS Total_Shipments,
    SUM(d.Delivery_Delayed) AS Delayed_Shipments,
    ROUND(AVG(d.Delivery_Delayed) * 100, 2) AS Delay_Rate_Pct,
    ROUND(AVG(t.Waiting_Time), 2) AS Avg_Waiting_Hours,
    ROUND(AVG(d.Actual_Transit_Days), 2) AS Avg_Transit_Days
FROM Transportation t
INNER JOIN Shipments s ON t.Shipment_ID = s.Shipment_ID
INNER JOIN Delivery_Records d ON s.Shipment_ID = d.Shipment_ID
GROUP BY t.Weather_Condition
ORDER BY Delay_Rate_Pct DESC;


-- ------------------------------------------------------------------------------
-- QUERY 8: Top 5 Highest Risk Logistics Lanes (Origin to Destination)
-- Business Value: Highlights routes with highest frequency of customer disappointment.
-- ------------------------------------------------------------------------------
SELECT 
    CONCAT(s.Origin, ' -> ', s.Destination) AS Shipping_Lane,
    COUNT(s.Shipment_ID) AS Volume,
    ROUND(AVG(s.Distance), 0) AS Distance_KM,
    ROUND(AVG(t.Transportation_Cost), 2) AS Avg_Cost,
    SUM(d.Delivery_Delayed) AS Delayed_Count,
    ROUND(AVG(d.Delivery_Delayed) * 100, 2) AS Lane_Delay_Rate_Pct
FROM Shipments s
INNER JOIN Transportation t ON s.Shipment_ID = t.Shipment_ID
INNER JOIN Delivery_Records d ON s.Shipment_ID = d.Shipment_ID
GROUP BY s.Origin, s.Destination
HAVING COUNT(s.Shipment_ID) >= 50
ORDER BY Lane_Delay_Rate_Pct DESC
LIMIT 5;


-- ------------------------------------------------------------------------------
-- QUERY 9: Warehouse Dock Congestion: Demurrage Impact (> 4 Hours vs <= 4 Hours)
-- Business Value: Proves correlation between warehouse loading delays and final delivery failure.
-- ------------------------------------------------------------------------------
SELECT 
    CASE 
        WHEN t.Waiting_Time > 4.0 THEN 'High Congestion (> 4 Hours)' 
        ELSE 'Normal Processing (<= 4 Hours)' 
    END AS Dock_Congestion_Tier,
    COUNT(s.Shipment_ID) AS Shipment_Count,
    ROUND(AVG(t.Waiting_Time), 2) AS Avg_Waiting_Hours,
    ROUND(AVG(t.Transportation_Cost), 2) AS Avg_Transportation_Cost,
    ROUND(AVG(d.Delivery_Delayed) * 100, 2) AS Delay_Rate_Pct,
    ROUND(AVG(d.Delivery_Delay_Days), 2) AS Avg_Days_Late
FROM Transportation t
INNER JOIN Shipments s ON t.Shipment_ID = s.Shipment_ID
INNER JOIN Delivery_Records d ON s.Shipment_ID = d.Shipment_ID
GROUP BY Dock_Congestion_Tier;


-- ------------------------------------------------------------------------------
-- QUERY 10: Machine Learning Audit: Prediction Accuracy & False Negative Assessment
-- Business Value: Measures in-database classifier precision, recall, and costly False Negatives.
-- ------------------------------------------------------------------------------
SELECT 
    p.Model_Used,
    COUNT(p.Prediction_ID) AS Total_Predictions,
    SUM(CASE WHEN p.Predicted_Status = d.Delivery_Status THEN 1 ELSE 0 END) AS Correct_Predictions,
    ROUND(AVG(CASE WHEN p.Predicted_Status = d.Delivery_Status THEN 1.0 ELSE 0.0 END) * 100, 2) AS Model_Accuracy_Pct,
    -- Confusion Matrix Elements:
    SUM(CASE WHEN p.Predicted_Status = 'Delayed' AND d.Delivery_Status = 'Delayed' THEN 1 ELSE 0 END) AS True_Positives,
    SUM(CASE WHEN p.Predicted_Status = 'On Time' AND d.Delivery_Status = 'On Time' THEN 1 ELSE 0 END) AS True_Negatives,
    SUM(CASE WHEN p.Predicted_Status = 'Delayed' AND d.Delivery_Status = 'On Time' THEN 1 ELSE 0 END) AS False_Positives,
    SUM(CASE WHEN p.Predicted_Status = 'On Time' AND d.Delivery_Status = 'Delayed' THEN 1 ELSE 0 END) AS False_Negatives_Costly
FROM Prediction_Results p
INNER JOIN Delivery_Records d ON p.Shipment_ID = d.Shipment_ID
GROUP BY p.Model_Used;


-- ------------------------------------------------------------------------------
-- QUERY 11: High Probability Risk Buffer: Shipments with Delay Probability >= 70%
-- Business Value: Automated dispatch trigger for proactive customer notifications or carrier rerouting.
-- ------------------------------------------------------------------------------
SELECT 
    s.Shipment_ID,
    s.Origin,
    s.Destination,
    s.Shipping_Mode,
    t.Weather_Condition,
    t.Traffic_Level,
    t.Waiting_Time,
    p.Delay_Probability,
    p.Predicted_Status,
    d.Delivery_Status AS Actual_Status
FROM Shipments s
INNER JOIN Transportation t ON s.Shipment_ID = t.Shipment_ID
INNER JOIN Prediction_Results p ON s.Shipment_ID = p.Shipment_ID
INNER JOIN Delivery_Records d ON s.Shipment_ID = d.Shipment_ID
WHERE p.Delay_Probability >= 0.70
ORDER BY p.Delay_Probability DESC
LIMIT 10;


-- ------------------------------------------------------------------------------
-- QUERY 12: Estimated Financial Loss from Delivery Delays
-- Business Value: Converts operational delays into CFO-ready financial liability figures ($50/day penalty).
-- ------------------------------------------------------------------------------
SELECT 
    s.Shipping_Mode,
    SUM(d.Delivery_Delayed) AS Total_Delayed_Shipments,
    SUM(d.Delivery_Delay_Days) AS Total_Days_Late,
    ROUND(SUM(d.Delivery_Delay_Days * 50.00), 2) AS Estimated_SLA_Penalty_USD,
    ROUND(SUM(t.Transportation_Cost), 2) AS Total_Spend_USD,
    ROUND((SUM(d.Delivery_Delay_Days * 50.00) / SUM(t.Transportation_Cost)) * 100, 2) AS Penalty_Cost_Ratio_Pct
FROM Shipments s
INNER JOIN Delivery_Records d ON s.Shipment_ID = d.Shipment_ID
INNER JOIN Transportation t ON s.Shipment_ID = t.Shipment_ID
GROUP BY s.Shipping_Mode
ORDER BY Estimated_SLA_Penalty_USD DESC;

