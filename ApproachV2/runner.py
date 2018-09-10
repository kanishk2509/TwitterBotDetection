import csv
import pandas as pd
from get_api import get_api
from get_tweet_ratios import get_data
from classifier import Classifier
from sklearn import datasets
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
    # Use the 1st file path when running remotely from other machine
    # file_path = 'https://raw.githubusercontent.com/kanishk2509/TwitterBotDetection/master/kaggle_data/' \
    #            'training_data_2_csv_UTF.csv'
    file_path = '/Users/kanishksinha/PycharmProjects/TwitterBotDetection/kaggle_data/training_data_2_csv_UTF.csv'
    training_data = pd.read_csv(file_path, encoding='utf-8')
    bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                       r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                       r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                       r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'

    training_data['screen_name_binary'] = training_data.screen_name.str.contains(bag_of_words_bot, case=False, na=False)
    training_data['name_binary'] = training_data.name.str.contains(bag_of_words_bot, case=False, na=False)
    training_data['description_binary'] = training_data.description.str.contains(bag_of_words_bot, case=False, na=False)
    training_data['status_binary'] = training_data.status.str.contains(bag_of_words_bot, case=False, na=False)
    training_data['listed_count_binary'] = (training_data.listed_count > 20000) == False
    # Extracting Features
    features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified',
                'followers_count',
                'friends_count', 'statuses_count', 'listed_count_binary', 'bot']
    X = training_data[features].iloc[:, :-1]
    y = training_data[features].iloc[:, -1]
    return X, y


def lookup(user_id):
    api = get_api(key[0], key[1], key[2], key[3])
    X = get_data(user_id, api)
    return X


def main():
    twitter_user_name = sys.argv[1].lstrip().rstrip()
    # Read the learned classifier from the file system, or create it if it does not exist
    file_name = './learned_classifier.p'
    rfc = Classifier()
    if os.path.isfile(file_name):
        rfc.import_from_file(file_name)
        print("\nClassifier loaded")
    else:
        # Learn the classifier
        # Extract our features and class label from the raw data
        X, y = get_training_data()
        # split data into training and test data
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
        print("\nTraining classifier...")
        rfc.learn(x_train, y_train, 44)
        rfc.export(file_name)  # save the classifier to disk

        # predict
        predicted_class_labels = rfc.predict(x_test)

        # calculate the accuracy of the classifier
        accuracy = rfc.get_classifier_accuracy(predicted_class_labels, y_test)
        print("Classifier accuracy: ", accuracy)

    # run user input through classifier
    print("Mining twitter data...")
    input_data = np.array(lookup(twitter_user_name)).reshape(1, -1)
    print(input_data)
    print("Done!")
    print("Predicting...")
    result = rfc.predict(input_data)
    print("Done!\n")
    if result[0] == 1:
        print("Your account is a bot!\n")
    else:
        print("Your account is a human!\n")


if __name__ == '__main__':
    main()
