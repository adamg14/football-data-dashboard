from data_exploration import match_data, numeric_data_frame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score, log_loss, ConfusionMatrixDisplay
import joblib

# import numerically encoded data frame from data_exploration module
# FEATURES / INPUT COLUMNS
X = numeric_data_frame.drop(["FTR", "FTAG", "FTHG"], axis=1)
y = numeric_data_frame["FTR"]

# split the data set into training and testing data
# 20% of the training data will be used for testing, the result will be used to train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# initialising/training a regression model to be able to predict the outcome of the match result based on the feature columns
# set the random seed
model = RandomForestClassifier(random_state=10)
model.fit(X_train, y_train)

# making predictions with the trained model on the testing data
y_prediction = model.predict(X_test)

# classification report
print(classification_report(y_test, y_prediction))

# accuracy of the model
accuracy = accuracy_score(y_test, y_prediction)
print(f"Accuracy: { accuracy }")

# compare the mean square error between the predicted result 
mse = mean_squared_error(y_test, y_prediction)
print(f"Mean Squared Error: { mse }")

# visualisation of incorrect predictions (misclassifications) through a confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

# save the so it does not need to be retrained each time it is used
joblib.dump(model, "./models/trained_classification_model.joblib")
print("machine learning model saved to: ./models/trained_classification_model.joblib")