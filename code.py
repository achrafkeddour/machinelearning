from sklearn.tree import DecisionTreeClassifier

# Step 1: Expanded Training Data (Features and Labels)
# Features: [Engine Size (cc), Number of Wheels, Weight (kg), Max Speed (km/h), Passenger Capacity]
features = [
    [1500, 4, 1200, 200, 5],   # Car (Sedan)
    [2000, 4, 1500, 220, 7],   # Car (SUV)
    [3000, 4, 2500, 180, 2],   # Truck (Light)
    [5000, 6, 8000, 140, 3],   # Truck (Heavy)
    [0, 2, 15, 30, 1],         # Bicycle
    [50, 2, 20, 40, 1],        # Bicycle (Electric)
    [125, 2, 180, 120, 2],     # Motorcycle (Standard)
    [250, 2, 200, 150, 2],     # Motorcycle (Sport)
    [600, 2, 230, 200, 2],     # Motorcycle (Heavy Bike)
    [3500, 4, 3000, 160, 8],   # Car (Truck-based SUV)
    [50, 3, 40, 20, 1],        # Tricycle
    [1800, 4, 1400, 210, 5],   # Car (Compact)
    [90, 2, 30, 25, 1],        # Bicycle (Road Bike)
    [800, 3, 400, 80, 3],      # Auto Rickshaw
    [2000, 4, 2000, 180, 12],  # Van (Passenger)
    [2500, 4, 2200, 170, 3],   # Pickup Truck
    [150, 2, 250, 100, 2],     # Motorcycle (Cruiser)
    [0, 4, 1200, 100, 5],      # Electric Car
    [300, 4, 1000, 110, 2],    # Quad Bike
    [3500, 8, 10000, 120, 50], # Bus
    [10000, 12, 18000, 110, 200], # Train
    [0, 4, 15, 25, 4],         # Pedal Car
    [7000, 10, 15000, 90, 40], # Heavy Construction Vehicle
    [1500, 6, 4000, 160, 10],  # Limousine
    [4500, 6, 7000, 120, 25],  # Minibus
    [8000, 18, 20000, 120, 3], # Semi-Truck
    [20, 2, 25, 45, 1],        # Electric Scooter
    [200, 3, 500, 50, 2],      # Tuk-Tuk
    [1200, 4, 1000, 170, 4],   # Sports Car
    [500, 2, 80, 60, 2],       # Moped
]
# Labels:
# 0 = Car, 1 = Motorcycle, 2 = Truck, 3 = Bicycle, 4 = Tricycle, 5 = Auto Rickshaw,
# 6 = Van, 7 = Pickup Truck, 8 = Electric Car, 9 = Quad Bike, 10 = Bus, 11 = Train,
# 12 = Pedal Car, 13 = Heavy Construction Vehicle, 14 = Limousine, 15 = Minibus,
# 16 = Semi-Truck, 17 = Electric Scooter, 18 = Tuk-Tuk, 19 = Sports Car, 20 = Moped
labels = [
    0, 0, 2, 2, 3, 3, 1, 1, 1, 0, 4, 0, 3, 5, 6, 7, 1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
]

# Step 2: Create and Train the Model
model = DecisionTreeClassifier()
model.fit(features, labels)

# Step 3: Accept User Input
print("Please enter the vehicle details for classification.")
engine_size = float(input("Enter the engine size (cc, use 0 for non-motorized vehicles): "))
num_wheels = int(input("Enter the number of wheels: "))
weight = float(input("Enter the weight of the vehicle (kg): "))
max_speed = float(input("Enter the maximum speed of the vehicle (km/h): "))
passenger_capacity = int(input("Enter the passenger capacity of the vehicle: "))

# Step 4: Make a Prediction
prediction = model.predict([[engine_size, num_wheels, weight, max_speed, passenger_capacity]])

# Step 5: Map Prediction to Vehicle Type
vehicle_types = {
    0: "Car",
    1: "Motorcycle",
    2: "Truck",
    3: "Bicycle",
    4: "Tricycle",
    5: "Auto Rickshaw",
    6: "Van",
    7: "Pickup Truck",
    8: "Electric Car",
    9: "Quad Bike",
    10: "Bus",
    11: "Train",
    12: "Pedal Car",
    13: "Heavy Construction Vehicle",
    14: "Limousine",
    15: "Minibus",
    16: "Semi-Truck",
    17: "Electric Scooter",
    18: "Tuk-Tuk",
    19: "Sports Car",
    20: "Moped",
}
result = vehicle_types[prediction[0]]
print(f"The vehicle is likely a: {result}")