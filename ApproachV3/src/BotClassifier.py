import pandas as pd
from GetApi import get_api
from GetAccountProperties import get_data
from classifiers.RForestClassifier import RFC
from classifiers.DTreeClassifier import DTC
from classifiers.MNBClassifier import MNB
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys

'''Twitter Account Keys'''
key = ['L5UQsE4pIb9YUJvP7HjHuxSvW',
       'HdRLPYgUqME194Wi2ThbbWfRd9BNHxIr2ms612oX9Yq1QXZdH7',
       '1039011019786342401-iDggGlhErT1KKdVGVXz4Kt7X8v0kIV',
       'MJ17S1uhCaI1zS3NBWksMaWdwjvAjn7cpji5vyhknfcUe']


def get_training_data():

    # Getting training data
    print("\nGetting the new Classifier data...")

    # Use the this file path when running remotely from other machine
    # file_path = 'https://raw.githubusercontent.com/kanishk2509/TwitterBotDetection/master/kaggle_data/' \
    #             'training_dataset_final.csv'

    # Use the this file path when running locally from personal machine for faster access
    file_path = '/Users/kanishksinha/PycharmProjects/TwitterBotDetection/ApproachV3/datasets/training_dataset_final.csv'
    training_data = pd.read_csv(file_path, encoding='utf-8-sig')

    # Replacing screen_name column to a binary value
    symbols = r'_|%|"| '
    training_data['screen_name_binary'] = training_data.screen_name.str.contains(symbols, case=False, na=False)

    # Extracting Features
    features = ['id', 'screen_name_binary', 'age', 'in_out_ratio', 'favorites_ratio', 'status_ratio',
                'account_rep', 'avg_tpd', 'hashtags_ratio', 'user_mentions_ratio',
                'mal_url_ratio', 'bot']
    X = training_data[features].iloc[:, :-1]
    y = training_data[features].iloc[:, -1]

    print(X, y)
    return X, y


def lookup(user_id):
    api = get_api(key[0], key[1], key[2], key[3])
    X = get_data(user_id, api)
    return X


def main():

    # Get 1st user input from command line. This is a twitter user id used to test the classifier
    twitter_user_name = sys.argv[1].lstrip().rstrip()

    # Get 2nd user input from command line. This is the type of classifier to train our system
    # RF => Random Forest
    # DT => Decision Tree
    # NB => Multinomial Naive Bayes
    classifier_type = sys.argv[2].lstrip().rstrip().lower()

    # Consult the trained classifier from the file system, or create it if it does not exist
    if classifier_type == 'rf':
        rfc = RFC()
        file_name = './rf_trained_classifier.p'
        if os.path.isfile(file_name):
            rfc.import_from_file(file_name)
            print("\nRandom Forest Classifier loaded")
        else:
            # Train the classifier
            # Extract the features and class label from the raw data
            X, y = get_training_data()

            # Split data into training and test data
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
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
        file_name = './dt_trained_classifier.p'
        if os.path.isfile(file_name):
            rfc.import_from_file(file_name)
            print("\nDecision Tree Classifier loaded")
        else:
            # Train the classifier
            # Extract the features and class label from the raw data
            X, y = get_training_data()

            # Split data into training and test data
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
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
        file_name = './nb_trained_classifier.p'
        if os.path.isfile(file_name):
            rfc.import_from_file(file_name)
            print("\nNaive Bayes Classifier loaded")
        else:
            # Train the classifier
            # Extract the features and class label from the raw data
            X, y = get_training_data()

            # Split data into training and test data
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
            print("\nTraining Naive Bayes Classifier...")
            rfc.learn(x_train, y_train)

            # Save the classifier to disk
            rfc.export(file_name)

            predicted_class_labels = rfc.predict(x_test)

            # Calculate the accuracy of the classifier
            accuracy = rfc.get_classifier_accuracy(predicted_class_labels, y_test)
            print("Classifier accuracy: ", accuracy)

    # Run user input through classifier
    print("Mining twitter data...")
    input_data = np.array(lookup(twitter_user_name)).reshape(1, -1)
    print(input_data)
    print("Mining Done!")
    print("\nClassifying user now...")

    # Predict the class Bot or Human for user id
    result = rfc.predict(input_data)
    print("\nClassification done!\n")
    print(result)
    if result[0] == 1:
        print("This account is a BOT!\n")
    else:
        print("This account is a HUMAN!\n")


if __name__ == '__main__':
    main()
