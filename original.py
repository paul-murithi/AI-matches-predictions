import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('datasets/results.csv')
data = pd.read_csv('datasets/results.csv').head(10000)
# Display the first few rows of the dataset
# print(data.head())
# Add a new column 'result' based on the comparison of home_score and away_score
data['result'] = 'Draw'
data.loc[data['home_score'] > data['away_score'], 'result'] = 'Home Win'
data.loc[data['home_score'] < data['away_score'], 'result'] = 'Away Win'

# Display the updated dataset
# print(data.head())

# Create dummy variables for the categorical 'result' column
data = pd.get_dummies(data, columns=['result'], drop_first=True)

# Create a function to calculate team win percentage
def calculate_win_percentage(team, data):
    total_matches = len(data)
    total_wins = data[data['home_team'] == team]['result_Home Win'].sum()

    if total_matches == 0:
        return 0.0

    win_percentage = total_wins / total_matches
    return win_percentage

# Recalculate win percentages
data['home_win_percentage'] = data['home_team'].apply(lambda x: calculate_win_percentage(x, data))
data['away_win_percentage'] = data['away_team'].apply(lambda x: calculate_win_percentage(x, data))

# Fill NaN values with 0
data.fillna(0, inplace=True)

# Reorder columns to have 'result_Home Win' at the end
data = data[['home_team', 'away_team', 'home_win_percentage', 'away_win_percentage', 'neutral', 'result_Home Win']]

# Display the updated dataset
# print(data.head())

# Select relevant features
features = ['home_win_percentage', 'away_win_percentage', 'neutral']
features += [col for col in data.columns if col.startswith('result_')]

# Create feature matrix X and target variable y
X = data[features]
y = data['result_Home Win']

#######
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
##
# Create a logistic regression model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train, y_train)
##########
# Make predictions on the testing set
y_pred = model.predict(X_test)

## Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Create LabelEncoder instance
label_encoder = LabelEncoder()


# Fit and transform the 'result_Home Win' column
data['outcome'] = label_encoder.fit_transform(data['result_Home Win'])


# Function to predict match outcome for home, away, and draw
# Function to predict match outcome for home, away, and draw
def predict_match_outcome(model, home_team, away_team, data):
    home_win_percentage = calculate_win_percentage(home_team, data)
    away_win_percentage = calculate_win_percentage(away_team, data)

    user_data = pd.DataFrame({
        'home_win_percentage': [home_win_percentage],
        'away_win_percentage': [away_win_percentage],
        'neutral': [False],  # Assuming matches are not neutral by default
        'result_Home Win': [False]  # Default value for the result
    })

    # Ensure the columns in user_data match the features used during training
    user_data = user_data[['home_win_percentage', 'away_win_percentage', 'neutral', 'result_Home Win']]

    # Make predictions using the model
    prediction_proba = model.predict_proba(user_data)

    # Display the predicted outcome probabilities
    class_labels = ['Away Win', 'Home Win']  # Map 'False' to 'Away Win' and 'True' to 'Home Win'
    print("Predicted Probabilities:")
    for label, prob in zip(class_labels, prediction_proba[0]):
        print(f"{label} Probability: {prob * 100:.2f}%")

# Example usage
home_team_input = input("Enter the home team: ").capitalize()
away_team_input = input("Enter the away team: ").capitalize()

predict_match_outcome(model, home_team_input, away_team_input, data)