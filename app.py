# Import required libraries
from sklearn.model_selection import train_test_split
from data_processing import preprocess_data, encode_outcome
from modelling import train_logistic_regression, evaluate_model, predict_match_outcome


def main():
    # dataset file path
    file_path = 'datasets/results.csv'
    data = preprocess_data(file_path)# data preprocesing
    data = encode_outcome(data)

    features = ['home_win_percentage', 'away_win_percentage', 'neutral']
    features += [col for col in data.columns if col.startswith('result_')]
    # Feature matrix X and target variable y
    X = data[features]
    y = data['result_Home Win']
    # dividing the data to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_logistic_regression(X_train, y_train)

    evaluate_model(model, X_test, y_test)
    # user input
    home_team_input = input("Enter the home team: ").capitalize()
    away_team_input = input("Enter the away team: ").capitalize()

    predict_match_outcome(model, home_team_input, away_team_input, data)

# Code to run the main project
if __name__ == "__main__":
    main()