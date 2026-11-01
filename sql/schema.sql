-- ==============================================================================
-- DATABASE SCHEMA: supply_chain_analytics
-- Supply Chain & Logistics Analytics with Explainable Delivery Delay Prediction
-- ==============================================================================

-- Drop database if exists to ensure clean idempotent environment
DROP DATABASE IF EXISTS supply_chain_analytics;
CREATE DATABASE supply_chain_analytics
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE supply_chain_analytics;

-- ==============================================================================
-- TABLE 1: Shipments (Core entity storing initial order & route parameters)
-- ==============================================================================
CREATE TABLE Shipments (
    Shipment_ID VARCHAR(20) NOT NULL,
    Order_Date DATE NOT NULL,
    Dispatch_Date DATE NOT NULL,
    Origin VARCHAR(50) NOT NULL,
    Destination VARCHAR(50) NOT NULL,
    Distance DECIMAL(8, 2) NOT NULL COMMENT 'Distance in kilometers',
    Shipping_Mode VARCHAR(30) NOT NULL,
    Dispatch_Latency_Days INT NOT NULL COMMENT 'Days between order and dispatch',
    Distance_Tier VARCHAR(30) NOT NULL,
    Is_Weekend_Order TINYINT(1) NOT NULL DEFAULT 0,
    Created_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT PK_Shipments PRIMARY KEY (Shipment_ID)
) ENGINE=InnoDB;

-- ==============================================================================
-- TABLE 2: Transportation (Operational environmental conditions & transit costs)
-- ==============================================================================
CREATE TABLE Transportation (
    Transportation_ID INT AUTO_INCREMENT NOT NULL,
    Shipment_ID VARCHAR(20) NOT NULL,
    Shipping_Mode VARCHAR(30) NOT NULL,
    Weather_Condition VARCHAR(30) NOT NULL,
    Traffic_Level VARCHAR(30) NOT NULL,
    Waiting_Time DECIMAL(5, 2) NOT NULL COMMENT 'Dock waiting time in hours',
    Transportation_Cost DECIMAL(10, 2) NOT NULL COMMENT 'Total transportation cost in USD',
    Cost_Per_KM DECIMAL(8, 3) NOT NULL COMMENT 'Cost efficiency per km',
    High_Waiting_Flag TINYINT(1) NOT NULL DEFAULT 0,
    CONSTRAINT PK_Transportation PRIMARY KEY (Transportation_ID),
    CONSTRAINT FK_Transportation_Shipment FOREIGN KEY (Shipment_ID) 
        REFERENCES Shipments(Shipment_ID) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;

-- ==============================================================================
-- TABLE 3: Delivery_Records (Post-delivery audit & final SLA outcomes)
-- ==============================================================================
CREATE TABLE Delivery_Records (
    Delivery_ID INT AUTO_INCREMENT NOT NULL,
    Shipment_ID VARCHAR(20) NOT NULL,
    Delivery_Date DATE NOT NULL,
    Actual_Transit_Days INT NOT NULL,
    Delivery_Status VARCHAR(20) NOT NULL COMMENT 'On Time or Delayed',
    Delivery_Delay_Days INT NOT NULL DEFAULT 0 COMMENT '0 if on time, >0 if delayed',
    Delivery_Delayed TINYINT(1) NOT NULL DEFAULT 0 COMMENT '0 = On Time, 1 = Delayed',
    CONSTRAINT PK_Delivery_Records PRIMARY KEY (Delivery_ID),
    CONSTRAINT FK_Delivery_Shipment FOREIGN KEY (Shipment_ID) 
        REFERENCES Shipments(Shipment_ID) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;

-- ==============================================================================
-- TABLE 4: Prediction_Results (Machine Learning Inference Log & Explainability)
-- ==============================================================================
CREATE TABLE Prediction_Results (
    Prediction_ID INT AUTO_INCREMENT NOT NULL,
    Shipment_ID VARCHAR(20) NOT NULL,
    Predicted_Status VARCHAR(20) NOT NULL COMMENT 'On Time or Delayed',
    Delay_Probability DECIMAL(5, 4) NOT NULL COMMENT 'Predicted probability [0.0000 - 1.0000]',
    Model_Used VARCHAR(50) NOT NULL COMMENT 'Logistic Regression or Decision Tree',
    Prediction_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Prediction_Correct TINYINT(1) NULL COMMENT 'Audited post-delivery accuracy flag',
    CONSTRAINT PK_Prediction_Results PRIMARY KEY (Prediction_ID),
    CONSTRAINT FK_Prediction_Shipment FOREIGN KEY (Shipment_ID) 
        REFERENCES Shipments(Shipment_ID) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;

-- ==============================================================================
-- INDEXES FOR HIGH-PERFORMANCE ANALYTICS
-- ==============================================================================
CREATE INDEX IDX_Shipments_Dates ON Shipments(Order_Date, Dispatch_Date);
CREATE INDEX IDX_Shipments_Route ON Shipments(Origin, Destination);
CREATE INDEX IDX_Transportation_Conditions ON Transportation(Weather_Condition, Traffic_Level);
CREATE INDEX IDX_Delivery_Status ON Delivery_Records(Delivery_Delayed, Delivery_Status);
CREATE INDEX IDX_Prediction_Prob ON Prediction_Results(Delay_Probability);

