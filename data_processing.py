# import the required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path, sample_size=35000): # Sample size reduce number of rows for more speed
    print('Reading the dataset....')
    data = pd.read_csv(file_path).head(sample_size)
    # New column result depending on the comparison of home_score and away_score
    data['result'] = 'Draw'
    data.loc[data['home_score'] > data['away_score'], 'result'] = 'Home Win'
    data.loc[data['home_score'] < data['away_score'], 'result'] = 'Away Win'

    # placeholder variables for the result column
    data = pd.get_dummies(data, columns=['result'], drop_first=True)

    # Function to calculate team win percentage
    def calculate_win_percentage(team, data):
        total_matches = len(data)
        total_wins = data[data['home_team'] == team]['result_Home Win'].sum()

        if total_matches == 0:
            return 0.0
        # calculate the winning percentage
        win_percentage = total_wins / total_matches
        return win_percentage

    # module to calculate win percentages again
    print('Calculating, please wait.....')
    data['home_win_percentage'] = data['home_team'].apply(lambda x: calculate_win_percentage(x, data))
    data['away_win_percentage'] = data['away_team'].apply(lambda x: calculate_win_percentage(x, data))

    # Fill empty values
    data.fillna(0, inplace=True)

    #Reorder columns
    data = data[['home_team', 'away_team', 'home_win_percentage', 'away_win_percentage', 'neutral', 'result_Home Win']]

    return data

def encode_outcome(data):
    label_encoder = LabelEncoder()
    data['outcome'] = label_encoder.fit_transform(data['result_Home Win'])
    return data
