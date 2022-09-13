import os
import pandas as pd
from fExtractor import FeatureExtractor_v2
import Classification
import numpy as np
import utils
import shutil


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
    #text_type = 'resumo', metodo='co-metrix', vigencia_23_24=True, network_features='local', use_embeddings=False)

    def classification(self):
        print('Classification task')
        con_publicaciones = (self.projects[self.projects['label'] == 1]).shape[0]
        sin_publicaciones = (self.projects[self.projects['label'] == 0]).shape[0]
        print('Publicaciones:', con_publicaciones, sin_publicaciones)
        number_of_samples = min(con_publicaciones, sin_publicaciones)
        print('Samples:', number_of_samples)
        print()

        #classifiers = ['DT', 'KNN', 'NB', 'SVM', 'RF', 'MLP']
        #classifiers = ['KNN', 'NB', 'SVM', 'RF', 'MLP']
        classifiers = ['KNN', 'NB']
        iterations = 2
        #limiars = [5, 10, 20, 50]
        limiars = [5]
        vsize = 300
        #number_features = [20, 50, 100, 200]
        #number_features = [20, 50, 100]
        number_features = [20, 50]

        final_res = ''

        if self.metodo == 'network':
            self.classification_network_features_v2(iterations, number_of_samples, classifiers,
                                                 limiars, vsize, number_features)
        else:
            final_res = self.classification_textual_features(iterations, number_of_samples, classifiers)

        shutil.rmtree(self.path_extra_folder)
        return final_res


    def auxiliar_classification(self):
        con_publicaciones = (self.projects[self.projects['label'] == 1.0]).shape[0]
        sin_publicaciones = (self.projects[self.projects['label'] == 0.0]).shape[0]
        print('Publicaciones:', con_publicaciones, sin_publicaciones)
        number_of_samples = min(con_publicaciones, sin_publicaciones)
        print('Samples:', number_of_samples)
        remove_stops = self.remove_stops

        balanced_df = self.projects.groupby('label', as_index=False, group_keys=False).apply(lambda s: s.sample(number_of_samples, replace=False))
        balanced_df.info()
        print(balanced_df.shape)

        number_features = [50]
        limiars = [5]
        word_embeddings = utils.get_w2v_embeddings(300)

        obj = FeatureExtractor_v2(method=self.metodo, text_type=self.text_type, corpus=balanced_df,
                                  co_measures=self.co_metrix_file, stops=remove_stops,
                                  limiars=limiars, word_features=number_features, word_embeddings=word_embeddings,
                                  extra_folder= self.path_extra_folder)
        features = obj.extract_features()

        for index, feature in enumerate(features):
            print(index, feature.shape)
            print(feature)

        shutil.rmtree(self.path_extra_folder)


    def classification_textual_features(self, iterations, number_of_samples, classifiers): #clasificationn
        print('Analyzing textual features')
        print()
        container = {i: [[] for _ in range(5)] for i in classifiers}
        methods = ['pre_1', 'rec_1', 'pre_0', 'rec_0', 'accuracy']
        for it in range(iterations):
            print('Iteration ' + str(it + 1))
            balanced_df = self.projects.groupby('label', as_index=False, group_keys=False).apply(lambda s: s.sample(number_of_samples, replace=False))
            balanced_df.info()
            print(balanced_df.shape)

            obj = FeatureExtractor_v2(method=self.metodo, text_type=self.text_type, corpus=balanced_df,
                                          co_measures=self.co_metrix_file, stops=self.remove_stops)
            features = obj.extract_features()
            Y = balanced_df["label"]
            print(features.shape)
            obj = Classification.Classification(classifiers, features, Y, class_report=True)
            scores = obj.get_scores()
            print(it + 1, scores)
            for c in scores:
                for i, score in enumerate(scores[c]):
                    container[c][i].append(score)
            print('End iterations\n\n')

        result_str = ''
        for c in container:
            print(c)
            print(container[c])
            for index, scores in enumerate(container[c]):
                avg = round(np.mean(scores), 2)
                std = round(np.std(scores), 2)
                result_str += str(avg) + '(+/-' + str(std) + '),'
                print(methods[index], ':', avg, std)
        result_str = result_str[:-1]
        print('Final results:',result_str)
        output_file = self.path_extra_folder[self.path_extra_folder.find('/') + 1:]
        output_file = 'results/' + output_file.replace('/', '.txt')
        print('output results:', output_file)
        print('\nHaberrrrrrr\n')
        file = open(output_file, 'w')
        file.write(result_str)
        file.close()
        return result_str


    def class_net_features_by_iteration(self, df, word_embeddings, limiars, number_features, classifiers):
        obj = FeatureExtractor_v2(method=self.metodo, text_type=self.text_type, corpus=df,
                                  co_measures=self.co_metrix_file, stops=self.remove_stops,
                                  word_embeddings=word_embeddings, limiars=limiars, word_features=number_features,
                                  extra_folder= self.path_extra_folder)
        features = obj.extract_features()
        Y = df["label"]

        results = [[] for _ in range(len(classifiers))]

        for index, feature_set in enumerate(features):
            feature_set = np.array(feature_set)
            obj = Classification.Classification(classifiers, feature_set, Y, class_report=True) # class_report : True
            scores = obj.get_scores() # {'SVM':[p1,r1,p0,r0,acc], 'NB':[p1,r1,p0,r0,acc]}
            print(index, feature_set.shape)
            print(scores)
            for i, score in enumerate(scores):
                results[i].append(score)
        print('Results classifiers:')
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
                print(index, score)
                container[index].append(score)
            print('End iterations\n\n')

        print('Final de finals')
        prueba = np.array(container)
        print(prueba.shape)
        print()

        final_results = []
        std_deviations = []

        for results in container:
            results = np.array(results)
            deviation = np.std(results, axis=0)
            deviation = [round(i, 2) for i in deviation]
            results = np.mean(results, axis=0)
            results = [round(i,2) for i in results]
            print(results)
            print(deviation)
            final_results.append(results)
            std_deviations.append(deviation)

        final_results = np.array(final_results)
        std_deviations = np.array(std_deviations)
        print('\n')
        #print(final_results)
        final_results = final_results.transpose()
        std_deviations = std_deviations.transpose()
        print('\n Transpuesta')
        print(final_results)
        print()
        print(std_deviations)

        output_file = self.path_extra_folder[self.path_extra_folder.find('/') + 1:]
        output_file = 'results/' + output_file.replace('/', '.txt')
        print('output results:', output_file)
        print('\nHaberrrrrrr\n')
        file = open(output_file, 'w')
        for mean, deviation in zip(final_results, std_deviations):
            str_res = ''
            for m,d in zip(mean, deviation):
                str_res+=str(m) + '(+/-' + str(d) + '),'
            str_res = str_res[:-1]
            print(str_res)
            file.write(str_res + '\n')
        file.close()


    def classification_network_features_v2(self, iterations, number_of_samples, classifiers, limiars, vsize, number_features):
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
            '''
            iteration_results = 
                [
                    [p1-p0-r1-r0-acc, p1-p0-r1-r0-acc, ... p1-p0-r1-r0-acc],  # SVM
                    [p1-p0-r1-r0-acc, p1-p0-r1-r0-acc, ... p1-p0-r1-r0-acc],  # DT
                    [p1-p0-r1-r0-acc, p1-p0-r1-r0-acc, ... p1-p0-r1-r0-acc]  # NB
                ]
            '''
            for index, score in enumerate(iteration_results):
                print(index, score)
                container[index].append(score)

        '''
        container = 
        [
          [[p1-p0-r1-r0-acc, ... p1-p0-r1-r0-acc], [p1-p0-r1-r0-acc, ... p1-p0-r1-r0-acc]],  # SVM
          [[p1-p0-r1-r0-acc, ... p1-p0-r1-r0-acc], [p1-p0-r1-r0-acc, ... p1-p0-r1-r0-acc]],  # DT
          [[p1-p0-r1-r0-acc, ... p1-p0-r1-r0-acc], [p1-p0-r1-r0-acc, ... p1-p0-r1-r0-acc]]  # NB
        ]
        '''
        print('\nTesting esto:')
        for result in container:
            print(len(result), len(result[0]), result)






if __name__ == '__main__':

    #obj = Manager(type_project='r', area='medicina', duracion='all-', metodo='freqs', text_type='title_assunto')
    #obj = Manager(type_project='r', area='medicina', duracion='all-', metodo='network', text_type='resumo', remove_stops=True)
    obj = Manager(type_project='d', area='medicina', duracion='all-', metodo='network',remove_stops=False)
    #obj = Manager(type_project='r', area='medicina', duracion='all-', metodo='co-metrix', remove_stops=True)
    obj.classification()
    #obj.auxiliar_classification()

