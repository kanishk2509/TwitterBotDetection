The project contains different approaches as mentioned in the proposal. Each ApproachVn directory is a stand alone implementation.

There is no need to make separate branches to implement a technique. Make a new ApproachVn directory and try to implement your logic there itself. We will decide what to merge and what to keep separated when we finalise the project. 

#Things to note :

1. Please do not modify any other ApproachVn directory apart from your own to avoid merge conflicts :D.

ApproachV1
----------
(Kanishk)

Bag of words approach, Sample implementation of Classifiers on some data.

ApproachV2 
----------
(Kanishk)

Bag of words approach, input a user id and the program will mine real time 
twitter data, send it to classifier and the classifier will tell if its a bot or not.

ApproachV3
----------
(Kanishk + Christopher)

**Implementation of Paper 1 of the proposal**

**One of the final approaches**

**Kanishk's Progress Details**

1. Implemented Step 3 and Step 4 of 1st paper mentioned in the proposal document.
2. Took the already labelled (bot or not) training dataset [a]. This is not the final training dataset.
3. For each twitter id, mined real time data with account properties component to generate a final training dataset [b].
4. Extracted features for training dataset [b].
5. Trained classifiers (DT, RF, MNB) with the above training dataset.
6. To test, input a twitter user id along with a classifier you like [c].
7. Program will mine real time twitter data for this test id.
8. Post mining, the program sends the above data to chosen classifiers. 
9. The classifier generates an output predicting whether the test id is a bot or human.

[a] : https://raw.githubusercontent.com/kanishk2509/TwitterBotDetection/master/kaggle_data/training_data_2_csv_UTF.csv

[b] : https://raw.githubusercontent.com/kanishk2509/TwitterBotDetection/master/kaggle_data/training_dataset_final.csv

[c] : For running step 6, navigate to 'ApproachVx/src' folder. Type the following command on the terminal:

python3 BotClassifier.py 'test user id' 'rf|dt|nb'

rf - Random Forest Classifier
dt - Decision Tree Classifier
nb - Naive Bayes Classifier

Example : 

python3 BotClassifier.py 16712547 rf

This will train the Classifier with Random Forest technique, if not already trained. Then it will mine 
the test user id provided and predict if its a bot or not. So far, its working fine with approx 89% accuracy.

**What Christopher needs to do now**

1. Step 1 and Step 2 of the 1st paper of the proposal.
2. I have made a directory with your name in the ApproachV3 dir. You can work in that easily or just paste your code there if you already did it.
3. Don't use any training or test dataset right now [1]
4. Just make a component that takes as input a twitter user id and outputs below 2 metrics

        a. Corrected Conditional Entropy
        b. Average Spam Probability 'avg[P(spam|M)]' : Avg Probability of all the user's tweets belonging to 'spam' class.  

5. You can use keys from keys.txt file or better, generate one of your own set of keys to avoid hitting too many requests.

[1] 

Since I(Kanishk) am in the process of mining more twitter data for greater accuracy, 
you do not need to create a training or test dataset for this.
I will take your component and will simply attach it with my Account Property Component 
to generate a proper training dataset once you are done.


#ApproachV4 : TBD
Sentiment approach of classifying bots as mentioned in 2nd paper of our proposal.
 
Please update this part with the progress, whenever started.

	

 


