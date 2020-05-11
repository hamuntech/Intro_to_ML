import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# save filepath to variable for easier access
melbourne_file_path = './input/melbourne-housing-snapshot/melb_data.csv'

# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

# print a summary of the data in Melbourne data
print(melbourne_data.describe())

#Display column names
print(melbourne_data.columns)

#Dropna drops missing values
melbourne_data = melbourne_data.dropna(axis=0)

#Selecting The Prediction Target
y = melbourne_data.Price
print("The snapshot of y: \n", y.head())

#Choosing "Features"
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

#Creating X DataFrame
X = melbourne_data[melbourne_features]
print("Features: \n" , X.describe())

#Data snapshot
print(X.head())

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

#Note that this is just a test as the predictions are made on new unknown data
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))




