import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from classifiers.RForestClassifier import RFC
from classifiers.DTreeClassifier import DTC
from classifiers.MNBClassifier import MNB
from sklearn.model_selection import train_test_split
import numpy as np
import os
from spam_metric.multinomial_bayes import load

'''Twitter Account Keys'''
key = ['L5UQsE4pIb9YUJvP7HjHuxSvW',
       'HdRLPYgUqME194Wi2ThbbWfRd9BNHxIr2ms612oX9Yq1QXZdH7',
       '1039011019786342401-iDggGlhErT1KKdVGVXz4Kt7X8v0kIV',
       'MJ17S1uhCaI1zS3NBWksMaWdwjvAjn7cpji5vyhknfcUe']

vectorizer, classifier = load()
test_size = 0.1
random_state = 0

symbols = ['Bot', 'bot', 'b0t', 'B0T', 'B0t', 'random', 'http', 'co', 'every', 'twitter', 'pubmed', 'news',
           'created', 'like', 'feed', 'tweet', 'tweeting', 'task', 'world', 'x', 'affiliated', 'latest', 'twitterbot',
           'project', 'botally', 'generated', 'image', 'reply', 'tinysubversions', 'biorxiv', 'digital', 'rt',
           'ckolderup', 'arxiv', 'rss', 'thricedotted', 'collection', 'want', 'backspace', 'maintained',
           'things', 'curated', 'see', 'us', 'people', 'every', 'love', 'please']

file_path = '/Users/kanishksinha/Desktop/TwitterBotDetection/Classifiers/final_merged.csv'


def get_training_data_feature():
    # Getting training data
    print("Training data with features...\n")

    # Feature engineering
    with \
            open(file_path,
                 'r+',
                 encoding="utf-8") as inp:
        reader = csv.DictReader(inp)

        arr = []

        for row in reader:
            if any(x in row['screen_name'].lower() for x in symbols):
                row['screen_name'] = float(True)
            else:
                row['screen_name'] = float(False)

            if any(x in row['description'].lower() for x in symbols):
                row['description'] = float(True)
            else:
                row['description'] = float(False)

            arr.append(row)

    # Extracting Features
    features = ['age', 'screen_name', 'in_out_ratio', 'favorites_ratio', 'status_ratio',
                'account_rep', 'avg_tpd', 'hashtags_ratio', 'user_mentions_ratio',
                'url_ratio', 'cce', 'spam_ratio', 'description', 'bot']

    training_data = pd.DataFrame(arr, columns=features)

    X = training_data[features].iloc[:, :-1]
    y = training_data[features].iloc[:, -1]

    print(X)

    return X, y


def get_training_data():
    # Getting training data
    print("Normal training data...\n")

    training_data = pd.read_csv(file_path, encoding='utf-8')

    # Extracting Features
    features = ['age', 'in_out_ratio', 'favorites_ratio', 'status_ratio',
                'account_rep', 'avg_tpd', 'hashtags_ratio', 'user_mentions_ratio',
                'url_ratio', 'cce', 'spam_ratio', 'bot']
    X = training_data[features].iloc[:, :-1]
    y = training_data[features].iloc[:, -1]

    return X, y


def get_test_data():

    test_file_path = 'https://raw.githubusercontent.com/kanishk2509/TwitterBotDetection/master/twitter_data/'
    test_dataframe = pd.read_csv(test_file_path + 'final_test_datasets/test-data-v3.csv')
    # Extracting Features
    features = ['id', 'age', 'in_out_ratio', 'favorites_ratio', 'status_ratio',
                'account_rep', 'avg_tpd', 'hashtags_ratio', 'user_mentions_ratio',
                'cce', 'spam_ratio', 'bot']

    X = test_dataframe[features].iloc[:, :-1]
    return X


def get_test_data_feature():
    test_file_path = '/Users/kanishksinha/Desktop/TwitterBotDetection/ApproachV3/temp_datasets/test-data-v3-new.csv'
    # Feature engineering
    with \
            open(test_file_path,
                 'r+',
                 encoding="utf-8") as inp:
        reader = csv.DictReader(inp)

        arr = []

        for row in reader:
            if any(x in row['screen_name'].lower() for x in symbols):
                row['screen_name'] = float(True)
            else:
                row['screen_name'] = float(False)

            if any(x in row['description'].lower() for x in symbols):
                row['description'] = float(True)
            else:
                row['description'] = float(False)

            arr.append(row)

    # Extracting Features
    features = ['id', 'age', 'screen_name', 'in_out_ratio', 'favorites_ratio', 'status_ratio',
                'account_rep', 'avg_tpd', 'hashtags_ratio', 'user_mentions_ratio',
                'cce', 'spam_ratio', 'description', 'bot']

    test_dataframe = pd.DataFrame(arr, columns=features)

    X = test_dataframe[features].iloc[:, :-1]

    return X


def train_classifiers(type):
    classifier_type = type.lstrip().rstrip().lower()
    X, y = get_training_data_feature()
    #X, y = get_training_data()

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
        file_name = './dt_trained_classifier.p'
        if os.path.isfile(file_name):
            rfc.import_from_file(file_name)
            print("\nDecision Tree Classifier loaded")
        else:
            # Train the classifier
            # Extract the features and class label from the raw data
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
        file_name = './nb_trained_classifier.p'
        if os.path.isfile(file_name):
            rfc.import_from_file(file_name)
            print("\nNaive Bayes Classifier loaded")
        else:
            # Train the classifier
            # Extract the features and class label from the raw data
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
        pd_test_data = get_test_data_feature()

        print("\nClassifying user now...")
        for i in pd_test_data.itertuples():
            data = np.array(i).reshape(1, -1)
            input_data = np.delete(data, np.s_[0:2], axis=1)
            result = rfc.predict(input_data)
            if result[0] is '1':
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

        print("\nClassification done and saved!\n")
    except TypeError as e:
        print(e)
        print('Please re-run the code with valid parameters')


if __name__ == '__main__':
    main()
