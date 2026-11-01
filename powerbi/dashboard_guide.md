# 📊 Power BI Dashboard Implementation Guide
## Project: Supply Chain & Logistics Analytics with Explainable Delivery Delay Prediction

---

## 📌 Executive Summary & Architecture
This guide provides complete, step-by-step instructions to assemble a four-page, executive-grade Power BI report using the clean datasets and DAX measures generated in this project.

### Data Model Architecture (Star Schema)
The reporting layer uses a Star Schema optimized for performance:
```
           +--------------------+
           |      Dim_Date      |
           +--------------------+
           | Date (PK)          |
           +---------+----------+
                     | 1
                     |
                     | * (Order_Date)
+--------------------+--------------------+
|               Fact_Shipments            |
+-----------------------------------------+
| Shipment_ID (PK)                        |
| Order_Date (FK)                         |
| Origin, Destination                     |
| Shipping_Mode (FK)                      |
| Weather_Condition, Traffic_Level        |
| Distance, Waiting_Time, Cost            |
| Delivery_Status, Delivery_Delay_Days    |
| Predicted_Status, Delay_Probability     |
+---------+--------------------+----------+
          | *                  | *
          |                    |
          | 1                  | 1
+---------+----------+   +-----+----------------+
|     Dim_Route      |   |  Dim_ShippingMode    |
+--------------------+   +----------------------+
| Route_Key (PK)     |   | Shipping_Mode (PK)   |
| Origin, Destination|   | SLA_Transit_Days     |
+--------------------+   +----------------------+
```

---

## 🚀 Phase 1: Ingestion & Relationship Setup

1. **Open Microsoft Power BI Desktop**.
2. Click **Get Data -> Text/CSV** and load all CSV files located in `data/powerbi/`:
   - `Fact_Shipments.csv`
   - `Dim_Date.csv`
   - `Dim_Route.csv`
   - `Dim_ShippingMode.csv`
   - `Model_Performance.csv`
3. In the **Model View**, verify the following 1-to-many (`1:*`) relationships with **Single Cross-Filter Direction**:
   - `Dim_Date[Date]` $\rightarrow$ `Fact_Shipments[Order_Date]`
   - `Dim_ShippingMode[Shipping_Mode]` $\rightarrow$ `Fact_Shipments[Shipping_Mode]`
   - `Dim_Route[Origin]` and `Destination` $\rightarrow$ `Fact_Shipments[Origin]` and `Destination`
4. **Apply Custom Theme**: Go to **View -> Themes -> Browse for themes**, select `powerbi/theme.json`.
5. **Create Measures Table**: Click **Enter Data**, name the table `_DAX_Measures`, and paste in the measures from `powerbi/dax_measures.txt`.

---

## 📑 Phase 2: Four-Page Dashboard Layout & Visual Construction

---

### 📄 Page 1: Executive Overview
*Target Audience: Chief Supply Chain Officer (CSCO) & VP of Logistics*

#### Top KPI Cards (Horizontal Banner):
1. **Total Shipments**: Metric `[Total Shipments]` (Formatted as `#,##0`).
2. **On-Time Delivery Rate (OTDR)**: Metric `[On-Time Delivery Rate (OTDR %)]` (Formatted as `0.0%`).
3. **Delayed Deliveries**: Metric `[Delayed Deliveries]` (Formatted as `#,##0`, Red color accent).
4. **Total Logistics Cost**: Metric `[Total Transportation Cost]` (Formatted as Currency `$#,##0`).
5. **Average Cost Per Shipment**: Metric `[Average Cost Per Shipment]` (Formatted as Currency `$#,##0.00`).

#### Visualizations:
1. **Monthly Volume & OTDR Trend (Combo Chart / Line & Stacked Column)**:
   - **X-Axis**: `Dim_Date[Year_Month]`
   - **Column Y-Axis**: `[Total Shipments]`
   - **Line Y-Axis**: `[On-Time Delivery Rate (OTDR %)]`
   - **Insight**: Displays seasonal order surges and detects whether volume spikes erode delivery punctuality.
2. **Shipment Volume by Shipping Mode (Donut Chart)**:
   - **Legend**: `Fact_Shipments[Shipping_Mode]`
   - **Values**: `[Total Shipments]`
   - **Data Label**: Value and percentage of total.
3. **Top 5 Origin Hubs Performance (Clustered Bar Chart)**:
   - **Y-Axis**: `Fact_Shipments[Origin]`
   - **X-Axis**: `[Total Shipments]`
   - **Color Gradient**: Ranked by `[Delivery Delay Rate (%)]`.

---

### 📄 Page 2: Delivery Performance & Bottleneck Analysis
*Target Audience: Regional Logistics Directors & Route Planners*

#### Top Filter / Slicers:
- **Origin Region** (Dropdown)
- **Destination Market** (Dropdown)
- **Shipping Mode** (Tiles)
- **Weather Condition** (Dropdown)

#### Visualizations:
1. **Origin-Destination Bottleneck Heatmap (Matrix Visual)**:
   - **Rows**: `Fact_Shipments[Origin]`
   - **Columns**: `Fact_Shipments[Destination]`
   - **Values**: `[Delivery Delay Rate (%)]`
   - **Conditional Formatting**: Background color gradient (Green 0% $\rightarrow$ Yellow 40% $\rightarrow$ Crimson Red 80%+).
   - **Takeaway**: Instantly spotlights specific lanes (e.g. Chicago $\rightarrow$ San Francisco) with chronic delays.
2. **Weather Impact on Delivery Delays (Clustered Column Chart)**:
   - **X-Axis**: `Fact_Shipments[Weather_Condition]` (Clear, Rain, Fog, Snow, Storm)
   - **Y-Axis**: `[Delivery Delay Rate (%)]` and `[Average Dock Waiting Time (hrs)]`
   - **Takeaway**: Demonstrates that storms and snow double the delay rate compared to clear skies.
3. **Traffic Level vs. Average Delay Duration (Clustered Bar Chart)**:
   - **Y-Axis**: `Fact_Shipments[Traffic_Level]` (Low, Medium, High, Severe)
   - **X-Axis**: `[Average Days Late]`
4. **Dock Waiting Time vs Delay Days (Scatter Plot)**:
   - **X-Axis**: `Fact_Shipments[Waiting_Time]`
   - **Y-Axis**: `Fact_Shipments[Delivery_Delay_Days]`
   - **Details**: `Fact_Shipments[Shipment_ID]`

---

### 📄 Page 3: Machine Learning & Explainable AI Analytics
*Target Audience: Lead Data Scientist, Analytics Manager & Operations Engineers*

#### Top Summary Banner:
- **Selected Champion Model**: Logistic Regression
- **Test Set Accuracy**: 74.2%
- **Test Set Precision**: 72.9%
- **Test Set Recall**: 68.0%
- **ROC-AUC Score**: 82.1%

#### Visualizations:
1. **Model Comparison Scorecard (Table Visual)**:
   - Source: `Model_Performance.csv`
   - Columns: `Model`, `Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`.
   - Takeaway: Shows why Logistic Regression was selected over the Decision Tree based on superior Recall and ROC-AUC.
2. **Top Delay Risk Drivers (Horizontal Bar Chart / Feature Importance)**:
   - **Y-Axis**: Top features from Logistic Regression (`Weather_Storm`, `Traffic_Severe`, `Waiting_Time`, `Distance`).
   - **X-Axis**: Coefficient Impact / Odds Ratio.
3. **Confusion Matrix Card Grid**:
   - 2x2 Matrix displaying:
     - Top-Left: `[True Negatives (TN)]` (Green)
     - Top-Right: `[False Positives (FP)]` (Amber)
     - Bottom-Left: `[False Negatives (FN - Costly SLA Breaches)]` (Red - Critical Alert)
     - Bottom-Right: `[True Positives (TP)]` (Blue)
4. **Delay Probability Distribution by Actual Status (Box Plot or Histogram)**:
   - **X-Axis**: `Fact_Shipments[Delivery_Status]`
   - **Y-Axis**: `Fact_Shipments[Delay_Probability]`
   - **Takeaway**: Shows clear separation where actual delayed shipments are assigned high probabilities by the model.

---

### 📄 Page 4: Business Insights & Strategic Action Plan
*Target Audience: Executive Leadership & Continuous Improvement Committee*

#### Visualizations:
1. **Root Cause Breakdown of Delays (Pareto Chart)**:
   - Bars: Volume of delayed shipments attributed to primary driver (Severe Weather, Dock Congestion >4h, Traffic Jam, Long-haul Rail).
   - Line: Cumulative percentage (80/20 rule).
2. **Financial SLA Penalty Exposure (Card Visuals)**:
   - Metric: `[Estimated SLA Delay Penalty]` (e.g., ~$150K liability).
   - Metric: `[Penalty Cost Ratio (%)]` (Percentage of total freight spend lost to delays).
3. **Strategic Recommendations Panel (Rich Text Card / Infographic)**:
   - **Action 1 (Dock Congestion):** Implement scheduled appointment windows for warehouses exceeding 4-hour waiting times.
   - **Action 2 (Weather Buffer):** Automatically add +1 day SLA buffer for shipments traversing winter snow corridors.
   - **Action 3 (High-Risk Carrier Shift):** Reallocate shipments on the top 3 bottleneck lanes from Standard Ground to expedited regional carriers.

---

## 🎛️ Interactive Global Slicers (Sync Slicers Enabled Across All Pages)
1. **Date Range**: `Dim_Date[Date]` (Between slicer).
2. **Origin Hub**: `Fact_Shipments[Origin]` (Multi-select dropdown).
3. **Shipping Mode**: `Fact_Shipments[Shipping_Mode]` (Horizontal pill buttons).
4. **Weather Condition**: `Fact_Shipments[Weather_Condition]` (Dropdown).
5. **Delivery Status**: `Fact_Shipments[Delivery_Status]` (On Time / Delayed toggle).

