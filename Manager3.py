
import pandas as pd
import os
import shutil
import utils
from fExtractor import FeatureExtractor
from Classification import Classification
import numpy as np

class Manager(object):

    def __init__(self, type_project='r', area='medicina', duracion='all', metodo='freqs', text_type='resumo', remove_stops=False):
        path_projects = 'datasets/projects/'
        path_doctorado = 'datasets/doctorado/'

        path_measures = 'datasets/measures/'
        path_doctorado_measures = 'datasets/measures_doc/'

        project_names = ['medicina_18_24.csv','odontologia_18_24.csv', 'veterinaria_18_24.csv',
                         'medicina_all.csv', 'genetica_all.csv', 'quimica_all.csv']

        self.metodo = metodo
        self.text_type = text_type
        self.remove_stops = remove_stops

        self.path_extra_folder = type_project + '_' + area + '_' + duracion + '_' + metodo + '_' + text_type + '_' + str(remove_stops)
        self.path_extra_folder = 'auxiliar_folder/' + self.path_extra_folder + '/'
        print(self.path_extra_folder)
        os.mkdir(self.path_extra_folder)
        path = ''
        path2 = ''
        if type_project == 'r':
            print(os.listdir(path_projects))
            if area == 'medicina':
                path = path_projects + project_names[0]
                path2 = path_measures + project_names[0]
            elif area == 'odontologia':
                path = path_projects + project_names[1]
                path2 = path_measures + project_names[1]
            elif area == 'veterinaria':
                path = path_projects + project_names[2]
                path2 = path_measures + project_names[2]
            projects = pd.read_csv(path)
            measures = pd.read_csv(path2)
            if duracion!='all':
                projects = projects.loc[projects["vigencia_months"]>=23]
        #elif type_project == 'd':
        else:
            print(os.listdir(path_doctorado))
            if area == 'medicina':
                path = path_doctorado + project_names[3]
                path2 = path_doctorado_measures + project_names[3]
            elif area == 'genetica':
                path = path_doctorado + project_names[4]
                path2 = path_doctorado_measures + project_names[4]
            elif area == 'quimica':
                path = path_doctorado + project_names[5]
                path2 = path_doctorado_measures + project_names[5]
            projects = pd.read_csv(path)
            measures = pd.read_csv(path2)
            if duracion != 'all':
                projects = projects.loc[(projects['vigencia_months'] >= 36) & (projects['vigencia_months'] <= 48)]

        self.projects = projects
        self.co_metrix_file = measures
        self.projects.info()
        print(self.projects.shape)

    def classification(self):
        print('Classification task')
        con_publicaciones = (self.projects[self.projects['label'] == 1]).shape[0]
        sin_publicaciones = (self.projects[self.projects['label'] == 0]).shape[0]
        print('Publicaciones:', con_publicaciones, sin_publicaciones)
        number_of_samples = min(con_publicaciones, sin_publicaciones)
        print('Samples:', number_of_samples)
        print()

        classifiers = ['KNN', 'NB', 'SVM', 'RF', 'MLP']
        #classifiers = ['KNN','SVM', 'RF']
        #classifiers = ['KNN', 'NB']
        iterations = 8
        limiars = [5]
        vsize = 300
        number_features = [50]

        final_res = ''

        if self.metodo == 'network':
            self.classification_network_features(iterations, number_of_samples, classifiers, limiars, vsize, number_features)
        else:
            final_res = self.classification_textual_features(iterations, number_of_samples, classifiers)

        shutil.rmtree(self.path_extra_folder)
        return final_res

    def classification_textual_features(self, iterations, number_of_samples, classifiers):
        print('Analyzing textual features')
        print()
        container = [[] for _ in range(len(classifiers))]
        for it in range(iterations):
            print('Iteration ' + str(it + 1))
            balanced_df = self.projects.groupby('label', as_index=False, group_keys=False).apply(lambda s: s.sample(number_of_samples, replace=False))
            balanced_df.info()
            print(balanced_df.shape)

            obj = FeatureExtractor(method=self.metodo, text_type=self.text_type, corpus=balanced_df,
                                      co_measures=self.co_metrix_file, stops=self.remove_stops)
            features = obj.extract_features()
            Y = balanced_df["label"]
            print(features.shape)
            obj = Classification(classifiers, features, Y)
            scores = obj.get_scores()
            for index, score in enumerate(scores):
                print(index, score)
                container[index].append(score)
            print('-------- End iteration' + str(it+1) + '------------------------\n\n')

        print('Test container:')
        # save results in txt!!!!!

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


    def class_net_features_by_iteration(self, df, word_embeddings, limiars, number_features, classifiers):
        obj = FeatureExtractor(method=self.metodo, text_type=self.text_type, corpus=df,
                                  co_measures=self.co_metrix_file, stops=self.remove_stops,
                                  word_embeddings=word_embeddings, limiars=limiars, word_features=number_features,
                                  extra_folder=self.path_extra_folder)

        features = obj.extract_features()
        Y = df["label"]
        results = [[] for _ in range(len(classifiers))]
        for index, feats in enumerate(features):
            print(index+1, feats.shape)
            obj = Classification(classifiers, feats, Y)
            scores = obj.get_scores()
            print(len(scores),scores)
            print()
            for i, score in enumerate(scores):
                results[i].extend(score)

        print('Results classifiers (5 results per classifiers):')
        for i in results:
            print(len(i), i)
            print()
        return results


    def classification_network_features(self, iterations, number_of_samples, classifiers, limiars, vsize, number_features):
        print('Analyzing network features')
        # load word embeddings
        print("Loading word embeddings!")
        word_embeddings = utils.get_w2v_embeddings(vsize)
        print('Total corpus words:', len(word_embeddings))
        container = [[] for _ in range(len(classifiers))]

        for it in range(iterations):
            print('Iteration ' + str(it + 1))
            balanced_df = self.projects.groupby('label', as_index=False, group_keys=False).apply(lambda s: s.sample(number_of_samples, replace=False))
            balanced_df.info()
            print(balanced_df.shape)

            iteration_results = self.class_net_features_by_iteration(balanced_df,
                                                                     word_embeddings, limiars,
                                                                     number_features, classifiers)

            for index, score in enumerate(iteration_results):
                container[index].append(score)
            print('-------- End iteration' + str(it+1) + '------------------------\n\n')

        final_results = []
        print('Testing result container:')
        for results in container:
            results = np.array(results)
            averages =  np.mean(results, axis=0)
            averages = [round(i, 2) for i in averages]
            deviation = np.std(results, axis=0)
            deviation = [round(i, 2) for i in deviation]
            str_results = utils.join_avg_devitations(averages, deviation)
            str_results = [str_results[i:i + 5] for i in range(0, len(str_results), 5)]
            final_results.append(str_results)
            print('Results:', len(str_results), str_results)
            print()

        print('\n\n')

        final_results = np.array(final_results)
        print('Haber normal', final_results.shape)

        output_file = self.path_extra_folder[self.path_extra_folder.find('/') + 1:]
        output_file = 'new_results/' + output_file.replace('/', '.txt')
        file = open(output_file, 'w')
        range_values = final_results.shape[1]
        for i in range(range_values):
            narray = final_results[:,i]
            result = (narray.reshape(-1)).tolist()
            result = ','.join(result)
            print(i + 1, result)
            file.write(result + '\n')
        file.close()

        print('Analisys finished')



if __name__ == '__main__':

    obj = Manager(type_project='d', area='medicina', duracion='all-', metodo='co-metrix', remove_stops=False)
    # obj = Manager(type_project='r', area='medicina', duracion='all-', metodo='freqs',remove_stops=False)##
    obj.classification()