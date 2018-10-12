import csv
import pandas as pd
from classifiers.RForestClassifier import RFC
from classifiers.DTreeClassifier import DTC
from classifiers.MNBClassifier import MNB
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

'''Twitter Account Keys'''
key = ['L5UQsE4pIb9YUJvP7HjHuxSvW',
       'HdRLPYgUqME194Wi2ThbbWfRd9BNHxIr2ms612oX9Yq1QXZdH7',
       '1039011019786342401-iDggGlhErT1KKdVGVXz4Kt7X8v0kIV',
       'MJ17S1uhCaI1zS3NBWksMaWdwjvAjn7cpji5vyhknfcUe']

symbols = r'_|%|"|nan| |Bot|bot|b0t|B0T|B0t|cannabis|tweet me|mishear|follow me|updates| ' \
          r'every|gorilla|yes_ofc|forget|FOLLOW|killin|genome|shout out|Save$$|Save $|' \
          r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA' \
          r'|jargon|Xanax|Only $|Free Instant|Extra income|Big bucks|$$$|Money making|' \
          r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|clone|genie|bbb|Viagra|' \
          r'ffd|emoji|Sale|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'

training_file_path = 'https://raw.githubusercontent.com/kanishk2509/TwitterBotDetection/master/twitter_data' \
                     '/final_training_datasets/training-dataset-final-v4.csv'

test_size = 0.1
random_state = 50


def get_training_data():
    # Getting training data
    print("\nGetting the training data...")

    # Use the this file path when running remotely from other machine
    file_path = training_file_path

    # Use the this file path when running locally from personal machine for faster access
    # file_path =
    # '/Users/kanishksinha/Desktop/TwitterBotDetection/kaggle_data/final_training_datasets/training-dataset' \
    # '-final-v4.csv'
    training_data = pd.read_csv(file_path, encoding='utf-8')

    # Feature engineering : Taking care of the incomplete/inappropriate/bot/spam data
    training_data['screen_name_binary'] = training_data.screen_name.str.contains(symbols, case=False, na=False)
    training_data['description_binary'] = training_data.description.str.contains(symbols, case=False, na=False)

    # Extracting Features
    features = ['id',
                'screen_name_binary',
                'age',
                'in_out_ratio',
                'favorites_ratio',
                'status_ratio',
                'account_rep',
                'avg_tpd',
                'hashtags_ratio',
                'user_mentions_ratio',
                'url_ratio',
                'avg_cosine_similarity',
                'avg_tweet_sentiment',
                'std_deviation_friends',
                'std_deviation_followers',
                'unique_urls_ratio',
                'tweet_url_similarity',
                'user_description_len',
                'user_description_sentiment',
                'special_char_in_description',
                'tweet_count',
                'description_binary',
                'bot']

    X = training_data[features].iloc[:, :-1]
    y = training_data[features].iloc[:, -1]

    return X, y


def get_test_data():
    file_path = 'https://raw.githubusercontent.com/kanishk2509/TwitterBotDetection/master/twitter_data/'
    test_dataframe = pd.read_csv(file_path + 'final_test_datasets/test-data-v4.csv')

    # Feature engineering
    test_dataframe['screen_name_binary'] = test_dataframe.screen_name.str.contains(symbols, case=False, na=False)
    test_dataframe['description_binary'] = test_dataframe.description.str.contains(symbols, case=False, na=False)
    # Extracting Features
    features = ['id',
                'screen_name_binary',
                'age',
                'in_out_ratio',
                'favorites_ratio',
                'status_ratio',
                'account_rep',
                'avg_tpd',
                'hashtags_ratio',
                'user_mentions_ratio',
                'url_ratio',
                'avg_cosine_similarity',
                'avg_tweet_sentiment',
                'std_deviation_friends',
                'std_deviation_followers',
                'unique_urls_ratio',
                'tweet_url_similarity',
                'user_description_len',
                'user_description_sentiment',
                'special_char_in_description',
                'tweet_count',
                'description_binary',
                'bot']

    X = test_dataframe[features].iloc[:, :-1]
    return X


def train_classifiers(type):
    path = '../ApproachV4/src/trained_classifiers/'
    classifier_type = type.lstrip().rstrip().lower()

    # Consult the trained classifier from the file system, or create it if it does not exist
    if classifier_type == 'rf':
        rfc = RFC()
        file_name = path + 'rf_trained_classifier.p'
        if os.path.isfile(file_name):
            rfc.import_from_file(file_name)
            print("\nRandom Forest Classifier loaded")
        else:
            # Train the classifier
            # Extract the features and class label from the raw data
            X, y = get_training_data()

            # Split data into training and test data
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            print("\nTraining Random Forest classifier...")
            rfc.learn(x_train, y_train, 22)

            # Save the classifier to disk
            rfc.export(file_name)

            predicted_class_labels = rfc.predict(x_test)

            # Calculate the accuracy of the classifier
            accuracy = rfc.get_classifier_accuracy(predicted_class_labels, y_test)
            print("Classifier accuracy: ", accuracy)

    elif classifier_type == 'dt':
        rfc = DTC()
        file_name = path + 'dt_trained_classifier.p'
        if os.path.isfile(file_name):
            rfc.import_from_file(file_name)
            print("\nDecision Tree Classifier loaded")
        else:
            # Train the classifier
            # Extract the features and class label from the raw data
            X, y = get_training_data()

            # Split data into training and test data
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            print("\nTraining Decision Tree Classifier...")
            rfc.learn(x_train, y_train)

            # Save the classifier to disk
            rfc.export(file_name)

            predicted_class_labels = rfc.predict(x_test)

            # Calculate the accuracy of the classifier
            accuracy = rfc.get_classifier_accuracy(predicted_class_labels, y_test)
            print("Classifier accuracy: ", accuracy)

    elif classifier_type == 'nb':
        rfc = MNB()
        file_name = path + 'nb_trained_classifier.p'
        if os.path.isfile(file_name):
            rfc.import_from_file(file_name)
            print("\nNaive Bayes Classifier loaded")
        else:
            # Train the classifier
            # Extract the features and class label from the raw data
            # Normalise negative values
            X, y = get_training_data()
            scaler = MinMaxScaler()
            print(scaler.fit(X))
            x = scaler.transform(X)
            # Split data into training and test data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
            print("\nTraining Naive Bayes Classifier...")
            rfc.learn(x_train, y_train)

            # Save the classifier to disk
            rfc.export(file_name)

            predicted_class_labels = rfc.predict(x_test)

            # Calculate the accuracy of the classifier
            accuracy = rfc.get_classifier_accuracy(predicted_class_labels, y_test)
            print("Classifier accuracy: ", accuracy)

    return rfc


def main():
    cl_type = 'rf'
    predicted_df = []
    try:
        # The program checks if the classifier is already trained. If not, trains again.
        rfc = train_classifiers(cl_type)

        print("\nGetting test data from repository...")
        pd_test_data = get_test_data()

        print("\nClassifying user now...")
        for i in pd_test_data.itertuples():
            data = np.array(i).reshape(1, -1)
            input_data = np.delete(data, 0, axis=1)
            result = rfc.predict(input_data)
            if result[0] == 1:
                dictn = {'id': i.id, 'bot': 1}
                predicted_df.append(dictn)
            else:
                dictn = {'id': i.id, 'bot': 0}
                predicted_df.append(dictn)

        with open('./classified_users.csv', 'w+', encoding="utf-8") as out:
            fields = ['id', 'bot']
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for row in predicted_df:
                writer.writerow({'id': row['id'],
                                 'bot': row['bot']})

        print("\nClassification done and saved in 'classified_users.csv'!\n")
    except TypeError as e:
        print(e)
        print('Please re-run the code with valid parameters')


if __name__ == '__main__':
    main()
