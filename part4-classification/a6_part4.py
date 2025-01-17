import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("part4-classification/suv_data.csv")
data['Gender'].replace(['Male', 'Female'], [0, 1], inplace=True)

x = data[["Age", "EstimatedSalary", "Gender"]].values
y = data["Purchased"].values

# Step 1: Print the values for x and y
print("X values (features):\n", x)
print("Y values (labels):\n", y)

# Step 2: Standardize the data using StandardScaler
scaler = StandardScaler()
scaler.fit(x)

# Step 3: Transform the data
x = scaler.transform(x)

# Step 4: Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Step 5: Fit the data

# Step 6: Create a LogisticRegression object and fit the data
model = linear_model.LogisticRegression()
model.fit(x_train, y_train)

# Step 7: Print the score to see the accuracy of the model
accuracy = model.score(x_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")
print("*************")
print("Testing Results:")

# Step 8: Print out the actual ytest values and predicted y values
# based on the xtest data
for i in range(len(x_test)):
    test_input = x_test[i].reshape(-1, 3)  # Reshape for prediction
    predicted_label = int(model.predict(test_input))

    # Convert predicted and actual labels to Male/Female
    predicted = "Male" if predicted_label == 0 else "Female"
    actual_label = y_test[i]
    actual = "Male" if actual_label == 0 else "Female"

    print(f"Predicted Gender: {predicted}, Actual Gender: {actual}")
    print("")
