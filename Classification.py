import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection as sk_ms
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

originalclass = []
predictedclass = []

def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred)


class Classification(object):

    def __init__(self, algorithms, features, labels, kfold=10, top_k_feats=10):
        self.features = features
        self.labels = labels
        self.kfold = kfold
        self.classifiers = algorithms
        self.top_k_feats = top_k_feats

    def get_scores_best_k(self):
        global originalclass
        global predictedclass
        #from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(self.features, self.labels)
        print(rf.feature_importances_)
        n = len(rf.feature_importances_)
        col_sorted_by_importance = (-rf.feature_importances_).argsort()[:n]
        col_sorted_by_importance = col_sorted_by_importance[0:self.top_k_feats]
        print(col_sorted_by_importance)
        top_features = self.features[:, col_sorted_by_importance]

        #rf = RandomForestClassifier(n_estimators=100)
        svm = SVC(gamma='auto')


        nested_score = sk_ms.cross_val_score(svm, top_features, self.labels, cv=self.kfold,
                                             scoring=make_scorer(classification_report_with_accuracy_score))
        report = classification_report(originalclass, predictedclass, output_dict=True)

        accuracy = report['accuracy']

        originalclass = []
        predictedclass = []

        accuracy = round(accuracy * 100, 4)
        return accuracy





    def get_scores(self):
        print('Classification')
        global originalclass
        global predictedclass
        class_dict = {'DT': DecisionTreeClassifier(random_state=0), 'KNN': KNeighborsClassifier(n_neighbors=3),
                      'NB': GaussianNB(), 'SVM': SVC(gamma='auto'), 'RF': RandomForestClassifier(n_estimators=100),
                      'MLP': MLPClassifier(random_state=0, max_iter=500)}

        results = []

        for algorithm in self.classifiers:
            print('Classification with ' + algorithm)
            nested_score = sk_ms.cross_val_score(class_dict[algorithm], self.features, self.labels, cv=self.kfold,
                                                 scoring=make_scorer(classification_report_with_accuracy_score))
            report = classification_report(originalclass, predictedclass, output_dict=True)

            #scores = sk_ms.cross_val_score(class_dict[algorithm], self.features, self.labels, cv=self.kfold,
            #                               scoring='accuracy', n_jobs=-1, verbose=0)
            
            #class_0_prec, class_0_rec = report['0.0']['precision'], report['0.0']['recall']
            #class_1_prec, class_1_rec = report['1.0']['precision'], report['1.0']['recall']

            precision = report['macro avg']['precision']
            recall = report['macro avg']['recall']
            f1 = report['macro avg']['f1-score']
            accuracy = report['accuracy']
            #accuracy = scores.mean()

            originalclass = []
            predictedclass = []

            #res = [class_1_prec * 100, class_1_rec * 100, class_0_prec * 100, class_0_rec * 100, accuracy * 100]
            #res = [round(i,4) for i in res]
            #results.append(res)

            res = [precision*100, recall*100, f1*100, accuracy*100]
            res = [round(i, 4) for i in res]
            results.append(res)
            #print(precision, recall, f1)

        return results

    def get_scores_voting(self):
        '''
        estimators = [('DT', DecisionTreeClassifier(random_state=0)), ('KNN', KNeighborsClassifier(n_neighbors=3)),
                      ('NB', GaussianNB()), ('SVM', SVC(gamma='auto')), ('RF', RandomForestClassifier(n_estimators=100)),
                      ('MLP', MLPClassifier(random_state=0, max_iter=500))]
        '''
        class_dict = {'DT': DecisionTreeClassifier(random_state=0), 'KNN': KNeighborsClassifier(n_neighbors=3),
                      'NB': GaussianNB(), 'SVM': SVC(gamma='auto', probability=True), 'RF': RandomForestClassifier(n_estimators=100),
                      'MLP': MLPClassifier(random_state=0, max_iter=500)}

        estimators = []
        for algorithm in self.classifiers:
            pair = (algorithm, class_dict[algorithm])
            estimators.append(pair)

        # create the ensemble model
        ensemble = VotingClassifier(estimators, voting='soft')
        results = sk_ms.cross_val_score(ensemble, self.features, self.labels, cv=10)
        return round(results.mean()*100, 4)



if __name__ == '__main__':

    # Voting Ensemble for Classification
    import pandas
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import VotingClassifier

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pandas.read_csv(url, names=names)
    array = dataframe.values
    X = array[:, 0:8]
    Y = array[:, 8]

    print(X.shape)
    print(Y.shape)

    #seed = 7
    #kfold = model_selection.KFold(n_splits=10, random_state=seed)
    #print(kfold)
    # create the sub models
    estimators = []
    model1 = LogisticRegression()
    estimators.append(('logistic', model1))
    model2 = DecisionTreeClassifier()
    estimators.append(('cart', model2))
    model3 = SVC()
    estimators.append(('svm', model3))
    #print(estimators)

    # create the ensemble model
    ensemble = VotingClassifier(estimators)
    results = model_selection.cross_val_score(ensemble, X, Y, cv=10)
    print(results)
    print(results.mean())

    '''
    - feat1, feat2, feat3
    '''

