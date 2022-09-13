import pandas as pd
import numpy as np
import os
#from new_feature_extractor import ResearcherFeatures, ResearcherManager
from researcherFeatures import ResearcherManager
from collections import Counter
from Classification import Classification
from fExtractor import FeatureExtractor
import utils
import matplotlib.pyplot as plt

class ProjectManager(object):

    def __init__(self, area='medicina', metodo='researcher', iterations=5,
                 last_years=False, by_area=False, top_words=20, features='all', ensemble=False, feature_selection=False):
        self.metodo = metodo
        self.area = area
        self.iterations = iterations
        self.ly = last_years
        self.by_area = by_area
        self.top_words = top_words
        self.top_k_feats = feature_selection
        size_dict = {'f1': 5, 'f2': 4, 'f3': 8, 'f4': 3, 'f5': 6, 'f6': 4, 'f7': top_words, 'f8': top_words * 2, 'f9': 2,
                     'f10': 1, 'f11': 5, 'f12': 0, 'f13': 8, 'f14': top_words * 2, 'f15': 7, 'f16': top_words * 2, 'f17': 0
                     }

        self.feature_names = []
        for feat in features:
            size = size_dict[feat]
            vector = [feat + '_' + str(x + 1) for x in range(size)]
            self.feature_names.extend(vector)


        if type(ensemble) is list:
            self.classifiers = ensemble
        else:
            self.classifiers = ['KNN', 'NB', 'SVM', 'RF', 'MLP']
        self.r_features = features
        self.use_ensemble = ensemble
        # classifiers = ['KNN', 'NB', 'SVM']
        if features == 'all' or features == 'f7' or 'f7' in features: #################
            self.flag_tfidf = True
        else:
            self.flag_tfidf = False


        path_projects = 'datasets/projects/'
        path_doctorado = 'datasets/doctorado/'

        print(os.listdir(path_projects))
        #path = path_projects + area + '_18_24.csv'
        path = path_projects + area + '.csv'
        #path2 = path_measures + project_names[2]
        projects = pd.read_csv(path)
        # ya actualize, no necesito mas hacer ese filtro
        #projects = projects[(projects['end_vigencia'] <= '2015-12-31')]
        #projects = projects.loc[projects["vigencia_months"] >= 23]

        self.projects = projects
        self.projects.info()

    def classification(self):
        print('Classification task')
        con_publicaciones = (self.projects[self.projects['label'] == 1]).shape[0]
        sin_publicaciones = (self.projects[self.projects['label'] == 0]).shape[0]
        print('Publicaciones:', con_publicaciones, sin_publicaciones)
        number_of_samples = min(con_publicaciones, sin_publicaciones)
        print('Samples:', number_of_samples)
        print()

        if self.metodo == 'researcher':
            final_res = self.classification_researcher_features(number_of_samples)
        else:
            final_res = ''
        return final_res

    def classification_researcher_features(self, number_of_samples):
        print('Analyzing researcher features')
        print()
        container = [[] for _ in range(len(self.classifiers))] #ly
        reManager = ResearcherManager(area=self.area,
                                      top_word=self.top_words, by_area=self.by_area,
                                      last_years=self.ly, flag_matrix=self.flag_tfidf)

        vect_parameters = [self.area, self.ly, self.r_features]#, self.top_words]
        single_results = []

        for it in range(self.iterations):
            print('Iteration ' + str(it + 1))
            balanced_df = self.projects.groupby('label', as_index=False, group_keys=False).apply(lambda s: s.sample(number_of_samples, replace=False))
            balanced_df.info()
            print(balanced_df.shape)

            #obj = FeatureExtractor(method=self.metodo, corpus=balanced_df, pesqObj=reManager, area=self.area,
            #                       last_years=self.ly, r_features=self.r_features)
            obj = FeatureExtractor(method=self.metodo, corpus=balanced_df, pesqObj=reManager,
                                   research_parameters=vect_parameters)
            df_features = obj.extract_features()
            print('Features de los balanceados:', df_features.shape)
            Y = balanced_df["label"]
            obj = Classification(self.classifiers, df_features, Y, top_k_feats=self.top_k_feats)
            print('Scoresss:')
            if type(self.use_ensemble) is list:
                score_ensemble = obj.get_scores_voting()
                print(score_ensemble)
                single_results.append(score_ensemble)

            elif type(self.top_k_feats) is int:
                score_top = obj.get_scores_best_k()
                print(score_top)
                single_results.append(score_top)

            else:
                scores = obj.get_scores()
                print(scores)

                for index, score in enumerate(scores):
                    print(index, score)
                    container[index].append(score)
            print('-------- End iteration' + str(it + 1) + '------------------------\n\n')

        if type(self.use_ensemble) is list or type(self.top_k_feats) is int:
            mean = round(np.mean(single_results),2)
            std = round(np.std(single_results),2)
            res = str(mean) + '(+/-' + str(std) + ')'
            print('Emsamble/Feat select:', single_results, mean, std, res)
            return res
        else:
            vector_results = []
            for results in container:
                results = np.array(results)
                averages = np.mean(results, axis=0)
                averages = [round(i, 2) for i in averages]
                deviation = np.std(results, axis=0)
                deviation = [round(i, 2) for i in deviation]
                str_results = utils.join_avg_devitations(averages, deviation)
                vector_results.extend(str_results)
                print(str_results)
                print()
            final_str = ','.join(vector_results)
            print('Final:', final_str)
            return final_str

    def feature_selection(self):
        print('Classification task')
        con_publicaciones = (self.projects[self.projects['label'] == 1]).shape[0]
        sin_publicaciones = (self.projects[self.projects['label'] == 0]).shape[0]
        print('Publicaciones:', con_publicaciones, sin_publicaciones)
        number_of_samples = min(con_publicaciones, sin_publicaciones)
        print('Samples:', number_of_samples)
        print()
        container = [[] for _ in range(len(self.classifiers))]  # ly
        reManager = ResearcherManager(area=self.area,
                                      top_word=self.top_words, by_area=self.by_area,
                                      last_years=self.ly, flag_matrix=self.flag_tfidf)

        vect_parameters = [self.area, self.ly, self.r_features]  # , self.top_words]
        voting_results = []

        balanced_df = self.projects.groupby('label', as_index=False, group_keys=False).apply(lambda s: s.sample(number_of_samples, replace=False))
        balanced_df.info()
        print(balanced_df.shape)


        obj = FeatureExtractor(method=self.metodo, corpus=balanced_df, pesqObj=reManager,
                               research_parameters=vect_parameters)
        df_features = obj.extract_features()
        print('Features de los balanceados:', df_features.shape)
        Y = balanced_df["label"]

        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(df_features, Y)
        print(rf.feature_importances_)
        n = len(rf.feature_importances_)
        col_sorted_by_importance = (-rf.feature_importances_).argsort()[:n]
        col_sorted_by_importance = col_sorted_by_importance[0:15]
        print('-----------')
        print(col_sorted_by_importance)
        print('-----------')
        prueba = df_features[:,col_sorted_by_importance]
        print(prueba)
        print(prueba.shape)
        print('-----------')


        print('names:', self.feature_names)
        feat_names = np.array(self.feature_names)
        feat_imp = pd.DataFrame({
            'cols': feat_names[col_sorted_by_importance],
            'imps': rf.feature_importances_[col_sorted_by_importance]
        })

        #import matplotlib.pyplot as plt
        #import plotly_express as px
        #px.bar(feat_imp, x='cols', y='imps')
        #plt.show()
        print(feat_imp)
        obj = Classification(self.classifiers, df_features, Y)
        print(obj.get_scores_best_k())



















if __name__ == '__main__':

    obj = ProjectManager(area='veterinaria', iterations=1, last_years=False)
    #obj.pruebita()
    obj.classification()

    '''
    type_project='r', area='medicina', metodo='researcher', iterations=5, 
    last_years=False, by_area=False, top_words=20
    '''

