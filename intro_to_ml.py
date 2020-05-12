import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

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
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']

#Creating X DataFrame
X = melbourne_data[melbourne_features]
print("Features: \n" , X.describe())

#Data snapshot
print(X.head())

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model1 = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model1.fit(X, y)

#Note that this is just a test as the predictions in real world scenarios are made on new unknown data
predicted_home_prices = melbourne_model1.predict(X)
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(predicted_home_prices)

#Absolute mean error
insample_error = mean_absolute_error(y, predicted_home_prices)
print("In-sample absolute mean error: ", insample_error)

#TRAINIG - VALIDATION DATA SPLIT

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model2 = DecisionTreeRegressor()
# Fit model
melbourne_model2.fit(train_X, train_y)

# print the top few validation predictions
print("Top few validation predictions: ", melbourne_model2.predict(val_X.head()))
# print the top few actual prices from validation data
print("Top few validation actual prices: ", val_y.head().tolist())

# get predicted prices on validation data
val_predictions = melbourne_model2.predict(val_X)
outsample_error = mean_absolute_error(val_y, val_predictions)
print("Out-of-sample absolute mean error: ", outsample_error)

#ERROR REDUCTION

#Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions, or
#Underfitting: failing to capture relevant patterns, again leading to less accurate predictions.

#Function to calculate Absolute mean error
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

#Use all examples with the best model with 500 nodes
final_model = DecisionTreeRegressor(max_leaf_nodes=500, random_state=1)
final_model.fit(X, y)
final_pred = final_model.predict(X)
final_mean_absolute_error = mean_absolute_error(y, final_pred)
print("Final In-sample Mean Absolute Error: ", final_mean_absolute_error)

#RANDOM FORESTS

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print("Forst Model Mean Absolute Error: ", mean_absolute_error(val_y, melb_preds))






