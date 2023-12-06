# import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# module to train a regression model on the given data.
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Use the trained model to predict results on the testing set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)
    return accuracy
# module to  Calculate the win percentage for a given team based on historical match data.
def calculate_win_percentage(team, data):
    # Total number of matches played by the team
    total_matches = len(data)
    total_wins = data[data['home_team'] == team]['result_Home Win'].sum()

    if total_matches == 0:# if a team has not played any matches
        return 0.0

    win_percentage = total_wins / total_matches
    return win_percentage

def predict_match_outcome(model, home_team, away_team, data):
    # Module to calculate win percentage for home and away teams using historical data
    home_win_percentage = calculate_win_percentage(home_team, data)
    away_win_percentage = calculate_win_percentage(away_team, data)

    user_data = pd.DataFrame({
        'home_win_percentage': [home_win_percentage],
        'away_win_percentage': [away_win_percentage],
        'neutral': [False],  
        'result_Home Win': [False] 
    })

    user_data = user_data[['home_win_percentage', 'away_win_percentage', 'neutral', 'result_Home Win']]

    # Make predictions using the model
    prediction_proba = model.predict_proba(user_data)

    # Display the predicted outcome probabilities
    class_labels = ['Away Win', 'Home Win']  # False is Away win, and True is Home Win
    print("Predicted Probabilities:")
    for label, prob in zip(class_labels, prediction_proba[0]):
        print(f"{label} Probability: {prob * 100:.2f}%")
