from sklearn.model_selection import train_test_split
import csv
import os
import pandas as pd
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def read_dataset():
    print("Normal dataset\n")
    bot_array = []
    user_array = []
    #file_path = '/home/chris/PycharmProjects/TwitterBotDetection/twitter_data/final_training_datasets/final_merged.csv'
    file_path = os.path.abspath('../twitter_data/final_training_datasets/final_merged.csv')
    with \
            open(file_path,
                 'r+',
                 encoding="utf-8") as inp:
        reader = csv.DictReader(inp)

        for row in reader:
            if row['bot'] == '1':
                array = [0 if float(row['age']) < 0 else float(row['age']),
                         0 if float(row['in_out_ratio']) < 0 else float(row['in_out_ratio']),
                         0 if float(row['favorites_ratio']) < 0 else float(row['favorites_ratio']),
                         0 if float(row['status_ratio']) < 0 else float(row['status_ratio']),
                         0 if float(row['account_rep']) < 0 else float(row['account_rep']),
                         0 if float(row['avg_tpd']) < 0 else float(row['avg_tpd']),
                         0 if float(row['hashtags_ratio']) < 0 else float(row['hashtags_ratio']),
                         0 if float(row['user_mentions_ratio']) < 0 else float(row['user_mentions_ratio']),
                         0 if float(row['url_ratio']) < 0 else float(row['url_ratio']),
                         0 if float(row['avg_cosine_similarity']) < 0 else float(row['avg_cosine_similarity']),
                         0 if float(row['avg_tweet_sentiment']) < 0 else float(row['avg_tweet_sentiment']),
                         0 if float(row['std_deviation_friends']) < 0 else float(row['std_deviation_friends']),
                         0 if float(row['std_deviation_followers']) < 0 else float(row['std_deviation_followers']),
                         0 if float(row['unique_urls_ratio']) < 0 else float(row['unique_urls_ratio']),
                         0 if float(row['tweet_url_similarity']) < 0 else float(row['tweet_url_similarity']),
                         0 if float(row['user_description_len']) < 0 else float(row['user_description_len'])]
                bot_array.append(deepcopy(array))

            else:
                array = [0 if float(row['age']) < 0 else float(row['age']),
                         0 if float(row['in_out_ratio']) < 0 else float(row['in_out_ratio']),
                         0 if float(row['favorites_ratio']) < 0 else float(row['favorites_ratio']),
                         0 if float(row['status_ratio']) < 0 else float(row['status_ratio']),
                         0 if float(row['account_rep']) < 0 else float(row['account_rep']),
                         0 if float(row['avg_tpd']) < 0 else float(row['avg_tpd']),
                         0 if float(row['hashtags_ratio']) < 0 else float(row['hashtags_ratio']),
                         0 if float(row['user_mentions_ratio']) < 0 else float(row['user_mentions_ratio']),
                         0 if float(row['url_ratio']) < 0 else float(row['url_ratio']),
                         0 if float(row['avg_cosine_similarity']) < 0 else float(row['avg_cosine_similarity']),
                         0 if float(row['avg_tweet_sentiment']) < 0 else float(row['avg_tweet_sentiment']),
                         0 if float(row['std_deviation_friends']) < 0 else float(row['std_deviation_friends']),
                         0 if float(row['std_deviation_followers']) < 0 else float(row['std_deviation_followers']),
                         0 if float(row['unique_urls_ratio']) < 0 else float(row['unique_urls_ratio']),
                         0 if float(row['tweet_url_similarity']) < 0 else float(row['tweet_url_similarity']),
                         0 if float(row['user_description_len']) < 0 else float(row['user_description_len'])]
                user_array.append(deepcopy(array))

    print(len(bot_array))
    print(len(user_array))
    features = user_array + bot_array[:len(user_array)]
    labels = ([0] * len(user_array)) + ([1] * len(user_array))

    return features, labels


def read_dataset_feature():
    print("Feature engineered dataset\n")
    symbols = ['Bot', 'bot', 'b0t', 'B0T', 'B0t', 'random', 'http', 'co', 'every', 'twitter', 'pubmed', 'news',
               'created', 'like', 'feed', 'tweeting', 'task', 'world', 'x', 'affiliated', 'latest', 'twitterbot',
               'project', 'botally', 'generated', 'image', 'reply', 'tinysubversions', 'biorxiv', 'digital', 'rt',
               'ckolderup', 'arxiv', 'rss', 'thricedotted', 'collection', 'want', 'backspace', 'maintained',
               'things', 'curated', 'see', 'us', 'people', 'every', 'love', 'please']
    bot_array = []
    user_array = []
    #file_path = '/home/chris/PycharmProjects/TwitterBotDetection/twitter_data/final_training_datasets/final_merged.csv'
    file_path = os.path.abspath('../twitter_data/final_training_datasets/final_merged.csv')
    with \
            open(file_path,
                 'r+',
                 encoding="utf-8") as inp:
        reader = csv.DictReader(inp)

        for row in reader:
            array_feature = [float(any(x in row['screen_name'] for x in symbols)),
                             float(any(x in row['description'] for x in symbols))]

            if row['bot'] == '1':
                array = [0 if float(row['age']) < 0 else float(row['age']),
                         0 if float(row['in_out_ratio']) < 0 else float(row['in_out_ratio']),
                         0 if float(row['favorites_ratio']) < 0 else float(row['favorites_ratio']),
                         0 if float(row['status_ratio']) < 0 else float(row['status_ratio']),
                         0 if float(row['account_rep']) < 0 else float(row['account_rep']),
                         0 if float(row['avg_tpd']) < 0 else float(row['avg_tpd']),
                         0 if float(row['hashtags_ratio']) < 0 else float(row['hashtags_ratio']),
                         0 if float(row['user_mentions_ratio']) < 0 else float(row['user_mentions_ratio']),
                         0 if float(row['url_ratio']) < 0 else float(row['url_ratio']),
                         0 if float(row['avg_cosine_similarity']) < 0 else float(row['avg_cosine_similarity']),
                         0 if float(row['avg_tweet_sentiment']) < 0 else float(row['avg_tweet_sentiment']),
                         0 if float(row['std_deviation_friends']) < 0 else float(row['std_deviation_friends']),
                         0 if float(row['std_deviation_followers']) < 0 else float(row['std_deviation_followers']),
                         0 if float(row['unique_urls_ratio']) < 0 else float(row['unique_urls_ratio']),
                         0 if float(row['tweet_url_similarity']) < 0 else float(row['tweet_url_similarity']),
                         0 if float(row['user_description_len']) < 0 else float(row['user_description_len']),
                         array_feature[0],
                         array_feature[1]]
                bot_array.append(deepcopy(array))

            else:
                array = [0 if float(row['age']) < 0 else float(row['age']),
                         0 if float(row['in_out_ratio']) < 0 else float(row['in_out_ratio']),
                         0 if float(row['favorites_ratio']) < 0 else float(row['favorites_ratio']),
                         0 if float(row['status_ratio']) < 0 else float(row['status_ratio']),
                         0 if float(row['account_rep']) < 0 else float(row['account_rep']),
                         0 if float(row['avg_tpd']) < 0 else float(row['avg_tpd']),
                         0 if float(row['hashtags_ratio']) < 0 else float(row['hashtags_ratio']),
                         0 if float(row['user_mentions_ratio']) < 0 else float(row['user_mentions_ratio']),
                         0 if float(row['url_ratio']) < 0 else float(row['url_ratio']),
                         0 if float(row['avg_cosine_similarity']) < 0 else float(row['avg_cosine_similarity']),
                         0 if float(row['avg_tweet_sentiment']) < 0 else float(row['avg_tweet_sentiment']),
                         0 if float(row['std_deviation_friends']) < 0 else float(row['std_deviation_friends']),
                         0 if float(row['std_deviation_followers']) < 0 else float(row['std_deviation_followers']),
                         0 if float(row['unique_urls_ratio']) < 0 else float(row['unique_urls_ratio']),
                         0 if float(row['tweet_url_similarity']) < 0 else float(row['tweet_url_similarity']),
                         0 if float(row['user_description_len']) < 0 else float(row['user_description_len']),
                         array_feature[0],
                         array_feature[1]]
                user_array.append(deepcopy(array))

    print(len(bot_array))
    print(len(user_array))
    features = user_array + bot_array[:len(user_array)]
    labels = ([0] * len(user_array)) + ([1] * len(user_array))

    return features, labels


def classify(train_features, train_labels, test_features, test_labels, randomstate=0):
    """
    This function classifies the features and labels on the Multinomial Naive Bayes, Random Forest and Decision Tree

    :param train_features: Training features
    :param train_labels: Training labels
    :param test_features: Test features
    :param test_labels: Test labels
    :param randomstate: Random state value to be passed to the decision tree and random forest for consistency in results

    :return: Accuracy value
    """

    clf_mnb = MultinomialNB(alpha=0.0009)
    clf_rf = RandomForestClassifier(random_state=randomstate)
    clf_dt = DecisionTreeClassifier(random_state=randomstate)

    clf_mnb.fit(train_features, train_labels)
    clf_rf.fit(train_features, train_labels)
    clf_dt.fit(train_features, train_labels)

    # Use features_test, test_transformed for testing with normal data or scaled data respectively
    predicted_mnb = clf_mnb.predict(test_features)
    predicted_rf = clf_rf.predict(test_features)
    predicted_dt = clf_dt.predict(test_features)

    accuracy_mnb = metrics.accuracy_score(test_labels, predicted_mnb)
    accuracy_rf = metrics.accuracy_score(test_labels, predicted_rf)
    accuracy_dt = metrics.accuracy_score(test_labels, predicted_dt)

    fpr_rf, tpr_rf, threshold_rf = roc_curve(test_labels, predicted_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    fpr_dt, tpr_dt, threshold_dt = roc_curve(test_labels, predicted_dt)
    roc_auc_dt = auc(fpr_dt, tpr_dt)
    fpr_nb, tpr_nb, threshold_nb = roc_curve(test_labels, predicted_mnb)
    roc_auc_nb = auc(fpr_nb, tpr_nb)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_rf, tpr_rf, 'b', label='Random Forest AUC = %0.2f' % roc_auc_rf)
    plt.plot(fpr_dt, tpr_dt, 'y', label='Decision Tree AUC = %0.2f' % roc_auc_dt)
    plt.plot(fpr_nb, tpr_nb, 'g', label='Naive Bayes AUC = %0.2f' % roc_auc_nb)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    return accuracy_mnb, accuracy_rf, accuracy_dt


def main():
    """
    In this function the prefix fe stands for feature engineered, mnb for multinomial bayes,
    rf for random forest, dt for decision tree. Scaled stands for variables related to scaled data.
    :return:
    """

    # Accuracy by classifier
    mnb_accuracy = []
    rf_accuracy = []
    dt_accuracy = []

    # accuracy arrays
    # no feature engineering with random state 0
    norm_feature_0 = []
    scaled_feature_0 = []

    norm_feature_53 = []
    scaled_feature_53 = []

    fe_norm_feature_0 = []
    fe_scaled_feature_0 = []

    fe_norm_feature_53 = []
    fe_scaled_feature_53 = []

    # Uncomment the followin line for using non feature engineering dataset
    features, labels = read_dataset()
    features_train, features_test, labels_train, labels_test = train_test_split(
        features,
        labels,
        test_size=0.1,  # use 10% for testing
        random_state=0)

    # Uncomment the following line for using feature engineered dataset
    fe_features, fe_labels = read_dataset_feature()

    fe_features_train, fe_features_test, fe_labels_train, fe_labels_test = train_test_split(
        fe_features,
        fe_labels,
        test_size=0.1,  # use 10% for testing
        random_state=0)

    scaler = MinMaxScaler()
    scaler.fit(features_train)
    transformed_train_features = scaler.transform(features_train)
    transformed_test_features = scaler.transform(features_test)

    fe_scaler = MinMaxScaler()
    fe_scaler.fit(fe_features)
    fe_transformed_train_features = fe_scaler.transform(fe_features_train)
    fe_transformed_test_features =fe_scaler.transform(fe_features_test)




    # Use features_train, transformed features for training with normal data or scaled data respectively

    accuracy_mnb, accuracy_rf, accuracy_dt = classify(features_train, labels_train,
                                                      features_test, labels_test,
                                                      randomstate=0)
    mnb_accuracy.append(accuracy_mnb)
    rf_accuracy.append(accuracy_rf)
    dt_accuracy.append(accuracy_dt)
    norm_feature_0.append(accuracy_mnb)
    norm_feature_0.append(accuracy_rf)
    norm_feature_0.append(accuracy_dt)
    print('Accuracy Naive Bayes with normal features and random state 0: ' + str(accuracy_mnb))
    print('Accuracy Random Forest with normal features and random state 0: ' + str(accuracy_rf))
    print('Accuracy Decision Tree with normal features and random state 0' + str(accuracy_dt))

    accuracy_mnb, accuracy_rf, accuracy_dt = classify(transformed_train_features, labels_train,
                                                      transformed_test_features, labels_test,
                                                      randomstate=0)
    mnb_accuracy.append(accuracy_mnb)
    rf_accuracy.append(accuracy_rf)
    dt_accuracy.append(accuracy_dt)
    scaled_feature_0.append(accuracy_mnb)
    scaled_feature_0.append(accuracy_rf)
    scaled_feature_0.append(accuracy_dt)
    print('Accuracy Naive Bayes with scaled features and random state 0: ' + str(accuracy_mnb))
    print('Accuracy Random Forest with scaled features and random state 0: ' + str(accuracy_rf))
    print('Accuracy Decision Tree with scaled features and random state 0: ' + str(accuracy_dt))


    accuracy_mnb, accuracy_rf, accuracy_dt = classify(features_train, labels_train,
                                                      features_test,  labels_test,
                                                      randomstate=53)
    mnb_accuracy.append(accuracy_mnb)
    rf_accuracy.append(accuracy_rf)
    dt_accuracy.append(accuracy_dt)
    norm_feature_53.append(accuracy_mnb)
    norm_feature_53.append(accuracy_rf)
    norm_feature_53.append(accuracy_dt)
    print('Accuracy Naive Bayes with normal features and random state 53: ' + str(accuracy_mnb))
    print('Accuracy Random Forest with normal features and random state 53: ' + str(accuracy_rf))
    print('Accuracy Decision Tree with normal features and random state 53' + str(accuracy_dt))

    accuracy_mnb, accuracy_rf, accuracy_dt = classify(transformed_train_features, labels_train,
                                                      transformed_test_features, labels_test,
                                                      randomstate=53)
    mnb_accuracy.append(accuracy_mnb)
    rf_accuracy.append(accuracy_rf)
    dt_accuracy.append(accuracy_dt)
    scaled_feature_53.append(accuracy_mnb)
    scaled_feature_53.append(accuracy_rf)
    scaled_feature_53.append(accuracy_dt)
    print('Accuracy Naive Bayes with scaled features and random state 53: ' + str(accuracy_mnb))
    print('Accuracy Random Forest with scaled features and random state 53: ' + str(accuracy_rf))
    print('Accuracy Decision Tree with scaled features and random state 53: ' + str(accuracy_dt))

    # Accuracies with feature engineered data
    accuracy_mnb, accuracy_rf, accuracy_dt = classify(fe_features_train, labels_train,
                                                      fe_features_test, labels_test,
                                                      randomstate=0)
    mnb_accuracy.append(accuracy_mnb)
    rf_accuracy.append(accuracy_rf)
    dt_accuracy.append(accuracy_dt)
    fe_norm_feature_0.append(accuracy_mnb)
    fe_norm_feature_0.append(accuracy_rf)
    fe_norm_feature_0.append(accuracy_dt)
    print('Accuracy Naive Bayes with feature engineering and random state 0: ' + str(accuracy_mnb))
    print('Accuracy Random Forest with feature engineering and random state 0: ' + str(accuracy_rf))
    print('Accuracy Decision Tree with feature engineering and random state 0' + str(accuracy_dt))

    accuracy_mnb, accuracy_rf, accuracy_dt = classify(fe_transformed_train_features, labels_train,
                                                      fe_transformed_test_features, labels_test,
                                                      randomstate=0)
    mnb_accuracy.append(accuracy_mnb)
    rf_accuracy.append(accuracy_rf)
    dt_accuracy.append(accuracy_dt)
    fe_scaled_feature_0.append(accuracy_mnb)
    fe_scaled_feature_0.append(accuracy_rf)
    fe_scaled_feature_0.append(accuracy_dt)
    print('Accuracy Naive Bayes with scaling and feature engineering and random state 0: ' + str(accuracy_mnb))
    print('Accuracy Random Forest with scaling and feature engineering and random state 0: ' + str(accuracy_rf))
    print('Accuracy Decision Tree with scaling and feature engineering and random state 0: ' + str(accuracy_dt))

    accuracy_mnb, accuracy_rf, accuracy_dt = classify(fe_features_train, labels_train,
                                                      fe_features_test, labels_test,
                                                      randomstate=53)
    mnb_accuracy.append(accuracy_mnb)
    rf_accuracy.append(accuracy_rf)
    dt_accuracy.append(accuracy_dt)
    fe_norm_feature_53.append(accuracy_mnb)
    fe_norm_feature_53.append(accuracy_rf)
    fe_norm_feature_53.append(accuracy_dt)
    print('Accuracy Naive Bayes with feature engineering and random state 53: ' + str(accuracy_mnb))
    print('Accuracy Random Forest with feature engineering and random state 53: ' + str(accuracy_rf))
    print('Accuracy Decision Tree with feature engineering and random state 53' + str(accuracy_dt))

    accuracy_mnb, accuracy_rf, accuracy_dt = classify(fe_transformed_train_features, labels_train,
                                                      fe_transformed_test_features, labels_test,
                                                      randomstate=53)
    mnb_accuracy.append(accuracy_mnb)
    rf_accuracy.append(accuracy_rf)
    dt_accuracy.append(accuracy_dt)
    fe_scaled_feature_53.append(accuracy_mnb)
    fe_scaled_feature_53.append(accuracy_rf)
    fe_scaled_feature_53.append(accuracy_dt)
    print('Accuracy Naive Bayes with scaling and feature engineering and random state 53: ' + str(accuracy_mnb))
    print('Accuracy Random Forest with scaling and feature engineering and random state 53: ' + str(accuracy_rf))
    print('Accuracy Decision Tree with scaling and feature engineering and random state 53: ' + str(accuracy_dt))

    # Pandas dataframes for displaying consolidated results
    print('========================ACCURACY GROUPED BY CLASSIFIER============================')
    d = {'MNB': mnb_accuracy, 'Random Forest': rf_accuracy, 'Decision Tree': dt_accuracy}
    df = pd.DataFrame(data=d)
    print(df)
    print('==================================================================================')

    print('========================ACCURACY GROUPED BY METHOD============================')
    d_transpose = {'Normal Data:RS-0': norm_feature_0, 'Scaled Data:RS-0': scaled_feature_0,
                   'Normal Data:RS-53': norm_feature_53, 'Scaled Data:RS-53': scaled_feature_53,
                   'FE Data:RS-0:': fe_norm_feature_0, 'FE Scaled Data:RS-0': fe_scaled_feature_0,
                   'FE Data:RS-53': fe_norm_feature_53, 'FE Scaled Data:RS-53': fe_scaled_feature_53}

    pd.set_option('display.max_columns', None)
    df_transpose = pd.DataFrame(data=d_transpose)
    print(df_transpose)
    print('==================================================================================')

    print('Row 0: MNB, Row 1: Random Forest, Row 2: Decision Tree')

if __name__ == '__main__':
    main()
