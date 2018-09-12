import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split

"""Performing Feature Engineering"""
print("Training the classifier, please wait ...")
print("\n")
filePath = 'https://raw.githubusercontent.com/kanishk2509/TwitterBotDetection/master/kaggle_data/' \
           'training_data_2_csv_UTF.csv'
training_data = pd.read_csv(filePath, encoding='utf-8')

bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                   r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                   r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                   r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'

training_data['screen_name_binary'] = training_data.screen_name.str.contains(bag_of_words_bot, case=False, na=False)
training_data['name_binary'] = training_data.name.str.contains(bag_of_words_bot, case=False, na=False)
training_data['description_binary'] = training_data.description.str.contains(bag_of_words_bot, case=False, na=False)
training_data['status_binary'] = training_data.status.str.contains(bag_of_words_bot, case=False, na=False)

"""Extracting Features"""
training_data['listed_count_binary'] = (training_data.listed_count > 20000) == False
features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified', 'followers_count',
            'friends_count', 'statuses_count', 'listed_count_binary', 'bot']

X = training_data[features].iloc[:, :-1]
y = training_data[features].iloc[:, -1]

print(X, y)

"""Decision Tree Classifier"""
dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=50, min_samples_split=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

dt = dt.fit(X_train, y_train)
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)

print("Decision Tree Classifier")
print("------------------------")
print("Training Accuracy: %.5f" % accuracy_score(y_train, y_pred_train))
print("Test Accuracy: %.5f" % accuracy_score(y_test, y_pred_test))
print("\n")

"""Multinomial Naive Bayes Classifier"""
mnb = MultinomialNB(alpha=0.0009)


mnb = mnb.fit(X_train, y_train)
y_pred_train = mnb.predict(X_train)
y_pred_test = mnb.predict(X_test)

print("Multinomial Naive Bayes Classifier")
print("----------------------------------")
print("Training Accuracy: %.5f" %accuracy_score(y_train, y_pred_train))
print("Test Accuracy: %.5f" %accuracy_score(y_test, y_pred_test))
print("\n")

"""Random Forest Classifier"""
rf = RandomForestClassifier(criterion='entropy', min_samples_leaf=100, min_samples_split=20)

rf = rf.fit(X_train, y_train)
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

print("Random Forest Classifier")
print("------------------------")
print("Trainig Accuracy: %.5f" %accuracy_score(y_train, y_pred_train))
print("Test Accuracy: %.5f" %accuracy_score(y_test, y_pred_test))
print("\n")
