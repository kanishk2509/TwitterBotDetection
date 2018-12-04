import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def feature_importance_rf_new(features_train, clf_rf):
    feat = pd.DataFrame(features_train)
    feat.columns = \
        ['age', 'in_out_ratio', 'favorites_ratio', 'status_ratio',
         'account_rep', 'avg_tpd', 'hashtags_ratio', 'user_mentions_ratio',
         'url_ratio', 'cce', 'spam_ratio', 'screen_name', 'description']

    feature_importance = pd.DataFrame(clf_rf.feature_importances_,
                                      index=feat.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)

    print(feature_importance)

    features = ['age', 'in_out_ratio', 'favorites_ratio', 'status_ratio',
                'account_rep', 'avg_tpd', 'hashtags_ratio', 'user_mentions_ratio',
                'url_ratio', 'cce', 'spam_ratio', 'screen_name', 'description']
    importance = clf_rf.feature_importances_
    indices = np.argsort(importance)

    plt.title('Feature Importance for Random Forest - New')
    plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


def feature_importance_rf_old(features_train, clf_rf):
    feat = pd.DataFrame(features_train)
    feat.columns = \
        ['age', 'in_out_ratio', 'favorites_ratio', 'status_ratio',
         'account_rep', 'avg_tpd', 'hashtags_ratio', 'user_mentions_ratio',
         'url_ratio', 'cce', 'spam_ratio']

    feature_importance = pd.DataFrame(clf_rf.feature_importances_,
                                      index=feat.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)

    print(feature_importance)

    features = ['age', 'in_out_ratio', 'favorites_ratio', 'status_ratio',
                'account_rep', 'avg_tpd', 'hashtags_ratio', 'user_mentions_ratio',
                'url_ratio', 'cce', 'spam_ratio']
    importance = clf_rf.feature_importances_
    indices = np.argsort(importance)

    plt.title('Feature Importance for Random Forest - Old')
    plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


def feature_importance_dt_new(features_train, clf_dt):
    feat = pd.DataFrame(features_train)
    feat.columns = \
        ['age', 'in_out_ratio', 'favorites_ratio', 'status_ratio',
         'account_rep', 'avg_tpd', 'hashtags_ratio', 'user_mentions_ratio',
         'url_ratio', 'cce', 'spam_ratio', 'screen_name', 'description']

    feature_importance = pd.DataFrame(clf_dt.feature_importances_,
                                      index=feat.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)

    print(feature_importance)

    features = ['age', 'in_out_ratio', 'favorites_ratio', 'status_ratio',
                'account_rep', 'avg_tpd', 'hashtags_ratio', 'user_mentions_ratio',
                'url_ratio', 'cce', 'spam_ratio', 'screen_name', 'description']
    importance = clf_dt.feature_importances_
    indices = np.argsort(importance)

    plt.title('Feature Importance for Decision Tree - New')
    plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


def feature_importance_dt_old(features_train, clf_dt):
    feat = pd.DataFrame(features_train)
    feat.columns = \
        ['age', 'in_out_ratio', 'favorites_ratio', 'status_ratio',
         'account_rep', 'avg_tpd', 'hashtags_ratio', 'user_mentions_ratio',
         'url_ratio', 'cce', 'spam_ratio']

    feature_importance = pd.DataFrame(clf_dt.feature_importances_,
                                      index=feat.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)

    print(feature_importance)

    features = ['age', 'in_out_ratio', 'favorites_ratio', 'status_ratio',
                'account_rep', 'avg_tpd', 'hashtags_ratio', 'user_mentions_ratio',
                'url_ratio', 'cce', 'spam_ratio']
    importance = clf_dt.feature_importances_
    indices = np.argsort(importance)

    plt.title('Feature Importance for Decision Tree - Old')
    plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
