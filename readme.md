The project contains different approaches as mentioned in the proposal. Each ApproachVn directory is a stand alone implementation.

ApproachV1
----------

Bag of words approach, Sample implementation of Classifiers on some data.

ApproachV2 
----------

Bag of words approach, input a user id and the program will mine real time 
twitter data, send it to classifier and the classifier will tell if its a bot or not.

ApproachV3
----------

**Implementation of Paper 1 of the proposal**

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


ApproachVx - Sentiment and Semantics
----------
(Kanishk)

1. Similar to ApproachV3
2. Computed these feature vectors (additional to the ones in ApproachV3) to feed into classifier
        
           a. "Average Tweet Sentiment" for all tweets belonging to a user
           b. "Average Tweet Similarity" between all tweets belonging to a user
	   c. "Average frequency of URLs" in all tweets belonging to a user
	  	d. Standard deviation of user's following
           e. Standard deviation of user's followers
           f. Average Frequency of @username mentioned in all the tweets
           g. Average number of "unique" urls in all the tweets, since bots tend to share similar urls over time.
           h. Length of user description
           i. Sentiment of user's description
           j. Total number of tweets
           
   After Christopher is done in ApproachV3, we will add these 2 features as well
           
           k. Corrected Conditional Entropy
           l. Average Spam Probability
                  

   For now, Don't use any training or test dataset 
    
   Just make a component that takes as input a twitter user id and computes the above features 

   This component will be simply appended with component mentioned in Point #2 to generate a proper training dataset.


Right now, accuracy of ApproachV3 is slightly better than this one.

For this approach, 2 papers have been consulted :

           a. Paper 2 of our proposal - "Using Sentiment to Detect Bots on Twitter: Are Humans more Opinionated than Bots?"
           b. A new paper - "Seven Months with the Devils: A Long-Term Study of Content Polluters on Twitter" 
                              Link: https://pdfs.semanticscholar.org/b433/9952a73914dc7eacf3b8e4c78ce9a5aa9502.pdf

This approach inclines towards implementation of Paper(b)
because exactly implementing and testing Paper(a) is too complicated 
since their dataset is huge and was collected for more than a month. 
Also, we don't have access to that dataset.


The approach mentioned in Paper(b) is good and implementable and helped in generating a credible training set.

	

 


