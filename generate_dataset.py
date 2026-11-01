"""
generate_dataset.py
-------------------
Generates a realistic, synthetic supply chain & logistics dataset for explainable
delivery delay prediction, meeting all project specifications:
- At least 5,000 records (generates 6,500 records)
- Realistic underlying physics & logistics behavior:
    * Distance & shipping mode determine expected baseline transit time.
    * Weather (Storm/Snow/Fog) and Traffic (Severe/High) dynamically elevate delay probability.
    * Extended warehouse dock waiting time directly impacts timely dispatch & transit.
    * Transportation cost reflects distance, mode premiums, and demurrage/waiting fees.
- Ground truth target 'Delivery_Delayed' (0 = On Time, 1 = Delayed) and 'Delivery_Delay_Days'.
- Full compliance with Data Leakage Prevention: Post-delivery outcomes are distinct.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_logistics_data(num_records: int = 6500, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    shipment_ids = [f"SHP-{10001 + i}" for i in range(num_records)]

    # Date range: 2023-01-01 to 2024-06-30 (18 months)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 30)
    date_range_days = (end_date - start_date).days

    random_days = np.random.randint(0, date_range_days, size=num_records)
    order_dates = [start_date + timedelta(days=int(d)) for d in random_days]

    # Major Origin hubs & Destination cities
    origins = ["Chicago", "Atlanta", "Dallas", "Los Angeles", "New York", "Seattle"]
    destinations = ["Miami", "Denver", "Boston", "Houston", "Phoenix", "San Francisco"]

    # Distance matrix approximation (km) between hubs
    base_distance_lookup = {
        ("Chicago", "Miami"): 2150, ("Chicago", "Denver"): 1620, ("Chicago", "Boston"): 1580,
        ("Chicago", "Houston"): 1750, ("Chicago", "Phoenix"): 2820, ("Chicago", "San Francisco"): 3430,
        ("Atlanta", "Miami"): 1070, ("Atlanta", "Denver"): 2260, ("Atlanta", "Boston"): 1720,
        ("Atlanta", "Houston"): 1280, ("Atlanta", "Phoenix"): 2910, ("Atlanta", "San Francisco"): 3920,
        ("Dallas", "Miami"): 2090, ("Dallas", "Denver"): 1280, ("Dallas", "Boston"): 2830,
        ("Dallas", "Houston"): 390, ("Dallas", "Phoenix"): 1720, ("Dallas", "San Francisco"): 2780,
        ("Los Angeles", "Miami"): 4390, ("Los Angeles", "Denver"): 1640, ("Los Angeles", "Boston"): 4800,
        ("Los Angeles", "Houston"): 2490, ("Los Angeles", "Phoenix"): 600, ("Los Angeles", "San Francisco"): 610,
        ("New York", "Miami"): 2070, ("New York", "Denver"): 2860, ("New York", "Boston"): 350,
        ("New York", "Houston"): 2620, ("New York", "Phoenix"): 3880, ("New York", "San Francisco"): 4680,
        ("Seattle", "Miami"): 5310, ("Seattle", "Denver"): 2100, ("Seattle", "Boston"): 4790,
        ("Seattle", "Houston"): 3750, ("Seattle", "Phoenix"): 2280, ("Seattle", "San Francisco"): 1300,
    }

    selected_origins = np.random.choice(origins, size=num_records, p=[0.20, 0.18, 0.18, 0.16, 0.16, 0.12])
    selected_destinations = np.random.choice(destinations, size=num_records, p=[0.18, 0.15, 0.16, 0.19, 0.16, 0.16])

    distances = []
    for o, d in zip(selected_origins, selected_destinations):
        base_dist = base_distance_lookup.get((o, d), 1800)
        # Add realistic route variation (+/- 8%)
        dist_variation = base_dist * np.random.uniform(0.92, 1.08)
        distances.append(round(dist_variation, 1))

    # Shipping Modes and baseline speeds
    shipping_modes = ["Standard Ground", "Express Air", "Same-Day Courier", "Freight Rail"]
    # Probability distribution across modes
    mode_choices = np.random.choice(
        shipping_modes,
        size=num_records,
        p=[0.45, 0.25, 0.10, 0.20]
    )

    # Weather conditions (seasonal probability)
    weather_options = ["Clear", "Rain", "Fog", "Snow", "Storm"]
    weather_choices = []
    for o_date in order_dates:
        # Winter months (Dec, Jan, Feb) have higher snow probability
        if o_date.month in [12, 1, 2]:
            w = np.random.choice(weather_options, p=[0.40, 0.20, 0.15, 0.18, 0.07])
        # Spring/Summer (May - Aug) have occasional storms/rain
        elif o_date.month in [5, 6, 7, 8]:
            w = np.random.choice(weather_options, p=[0.55, 0.25, 0.05, 0.00, 0.15])
        else:
            w = np.random.choice(weather_options, p=[0.60, 0.22, 0.10, 0.03, 0.05])
        weather_choices.append(w)

    # Traffic levels
    traffic_options = ["Low", "Medium", "High", "Severe"]
    traffic_choices = np.random.choice(
        traffic_options,
        size=num_records,
        p=[0.30, 0.40, 0.20, 0.10]
    )

    # Loading dock / warehouse waiting time in hours (0.5 to 14.0 hours, gamma distributed)
    raw_waiting = np.random.gamma(shape=2.5, scale=1.5, size=num_records)
    waiting_times = np.clip(np.round(raw_waiting + 0.5, 2), 0.5, 14.0)

    # Dispatch latency: 0 to 3 days depending on waiting time and order day
    dispatch_dates = []
    for o_date, wait in zip(order_dates, waiting_times):
        # Higher waiting time or weekend orders can add 1-2 days to dispatch
        extra_dispatch_days = 0
        if wait > 6.0:
            extra_dispatch_days += np.random.choice([1, 2], p=[0.7, 0.3])
        if o_date.weekday() >= 5:  # Weekend order
            extra_dispatch_days += 1
        dispatch_dates.append(o_date + timedelta(days=int(extra_dispatch_days)))

    # Transportation Cost model:
    # Base fee + per_km rate * distance + mode multiplier + waiting detention surcharge
    base_fee_map = {"Standard Ground": 50, "Express Air": 150, "Same-Day Courier": 120, "Freight Rail": 80}
    per_km_map = {"Standard Ground": 0.45, "Express Air": 1.20, "Same-Day Courier": 1.50, "Freight Rail": 0.25}

    transportation_costs = []
    for mode, dist, wait, traffic in zip(mode_choices, distances, waiting_times, traffic_choices):
        base_cost = base_fee_map[mode]
        distance_cost = dist * per_km_map[mode]
        detention_fee = max(0, (wait - 2.0)) * 25.0  # $25/hr after 2h free time
        traffic_surcharge = {"Low": 0, "Medium": 15, "High": 40, "Severe": 80}[traffic]
        noise = np.random.normal(0, 15)
        total_cost = base_cost + distance_cost + detention_fee + traffic_surcharge + noise
        transportation_costs.append(round(max(total_cost, 45.0), 2))

    # Underlying realistic Delay Probability Equation (Latent Risk Model):
    # Logit model representing realistic logistics vulnerabilities
    delay_flags = []
    delay_days_list = []
    delivery_dates = []

    # Standard expected transit days by mode & distance
    def calculate_expected_transit_days(mode, dist):
        if mode == "Same-Day Courier":
            return 1
        elif mode == "Express Air":
            return max(1, int(np.ceil(dist / 1800)))
        elif mode == "Standard Ground":
            return max(2, int(np.ceil(dist / 650)))
        elif mode == "Freight Rail":
            return max(3, int(np.ceil(dist / 450)))
        return 3

    for i in range(num_records):
        dist = distances[i]
        mode = mode_choices[i]
        weather = weather_choices[i]
        traffic = traffic_choices[i]
        wait = waiting_times[i]
        disp_date = dispatch_dates[i]
        order_date = order_dates[i]

        expected_transit = calculate_expected_transit_days(mode, dist)

        # Baseline delay probability z-score
        z = -2.2  # Base probability ~ 10% under ideal conditions

        # Impact of distance
        z += (dist / 1200) * 0.45

        # Impact of waiting time at warehouse/dock
        if wait > 7.0:
            z += 1.40
        elif wait > 4.0:
            z += 0.75
        elif wait > 2.5:
            z += 0.25

        # Impact of traffic
        traffic_weight = {"Low": -0.4, "Medium": 0.0, "High": 0.85, "Severe": 1.60}
        z += traffic_weight[traffic]

        # Impact of weather
        weather_weight = {"Clear": -0.5, "Rain": 0.35, "Fog": 0.65, "Snow": 1.30, "Storm": 1.85}
        z += weather_weight[weather]

        # Impact of shipping mode vulnerabilities:
        # Ground and Same-day are vulnerable to severe traffic
        if mode in ["Standard Ground", "Same-Day Courier"] and traffic in ["High", "Severe"]:
            z += 0.60
        # Express Air is vulnerable to Storms and Fog
        if mode == "Express Air" and weather in ["Storm", "Fog"]:
            z += 1.10
        # Freight rail is vulnerable to snow / track blockages
        if mode == "Freight Rail" and weather in ["Snow", "Storm"]:
            z += 0.80

        # Dispatch lead time impact (if dispatch took >1 day)
        disp_latency = (disp_date - order_date).days
        if disp_latency >= 2:
            z += 0.70

        # Compute probability via logistic sigmoid function
        prob_delay = 1.0 / (1.0 + np.exp(-z))

        # Sample actual delay status
        is_delayed = 1 if np.random.rand() < prob_delay else 0
        delay_flags.append(is_delayed)

        if is_delayed == 1:
            # Delay duration: 1 to 6 days depending on severity
            if weather in ["Storm", "Snow"] or traffic == "Severe":
                days_delayed = int(np.random.choice([2, 3, 4, 5, 6], p=[0.25, 0.35, 0.20, 0.12, 0.08]))
            elif wait > 6.0:
                days_delayed = int(np.random.choice([1, 2, 3, 4], p=[0.35, 0.40, 0.15, 0.10]))
            else:
                days_delayed = int(np.random.choice([1, 2, 3], p=[0.60, 0.30, 0.10]))
        else:
            days_delayed = 0

        delay_days_list.append(days_delayed)

        # Actual Delivery Date = Dispatch_Date + expected_transit + delay_days
        actual_transit_days = expected_transit + days_delayed
        actual_delivery = disp_date + timedelta(days=actual_transit_days)
        delivery_dates.append(actual_delivery)

    delivery_status = ["Delayed" if d == 1 else "On Time" for d in delay_flags]

    # Assemble complete DataFrame with exact required column names
    df = pd.DataFrame({
        "Shipment_ID": shipment_ids,
        "Order_Date": [d.strftime("%Y-%m-%d") for d in order_dates],
        "Dispatch_Date": [d.strftime("%Y-%m-%d") for d in dispatch_dates],
        "Delivery_Date": [d.strftime("%Y-%m-%d") for d in delivery_dates],
        "Origin": selected_origins,
        "Destination": selected_destinations,
        "Distance": distances,
        "Shipping_Mode": mode_choices,
        "Weather_Condition": weather_choices,
        "Traffic_Level": traffic_choices,
        "Waiting_Time": waiting_times,
        "Transportation_Cost": transportation_costs,
        "Delivery_Status": delivery_status,
        "Delivery_Delay_Days": delay_days_list,
        "Delivery_Delayed": delay_flags
    })

    return df

if __name__ == "__main__":
    df = generate_logistics_data(6500)
    output_path = "data/raw/supply_chain_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records successfully saved to {output_path}")
    print(f"Delay rate: {df['Delivery_Delayed'].mean() * 100:.2f}%")
    print(df.head())

