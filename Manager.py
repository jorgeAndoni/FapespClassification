
import utils
import pandas as pd
import fExtractor
import Classification
import numpy as np
from sklearn.preprocessing import StandardScaler

class Manager(object):

  def __init__(self, language='ptg', area='medicina', text_type = 'resumo', metodo='co-metrix', vigencia_23_24=True, network_features='local', use_embeddings=False):
    path = "datasets/"
    medicina_name = "medicina_18_24.csv"
    odontologia_name = "odontologia_18_24.csv"
    veterinaria_name = "veterinaria_18_24.csv"
    self.language = language


    if use_embeddings:
        print("Loading word embeddings!")
        self.word_embeddings = utils.get_w2v_vectors([path, medicina_name, odontologia_name, veterinaria_name])
        print('Total corpus words:', len(self.word_embeddings))
    else:
        self.word_embeddings = None

    self.metodo = metodo
    self.text_type = text_type
    self.network_features = network_features
    if area == 'medicina':
        area_csv = medicina_name
    elif area == 'odontologia':
        area_csv = odontologia_name
    elif area == 'veterinaria':
        area_csv = veterinaria_name
    else:
        area_csv = ''

    if self.language == 'ptg':
        lang = '/'
    else:
        lang = '_en/'

    file_projects = path + "projects" + lang + area_csv
    file_measures = path + "measures/" + area_csv
    projects_18_24 = pd.read_csv(file_projects)
    measures_18_24 = pd.read_csv(file_measures)
    if self.language == 'ptg':
        projects_18_24['index'] = list(measures_18_24["index"])
    projects_18_24.loc[projects_18_24['numero_publicaciones'] <= 0, 'label'] = 0
    projects_18_24.loc[projects_18_24['numero_publicaciones'] > 0, 'label'] = 1
    if not vigencia_23_24:  # para 18  a 24
        self.projects = projects_18_24
        self.measures = measures_18_24
    else:  # para 23 - 24
        self.projects = projects_18_24.loc[projects_18_24["vigencia_months"] >= 23]
        if self.language == 'ptg':
            indexes = list(self.projects['index'])
            self.measures = measures_18_24[measures_18_24['index'].isin(indexes)]
        else:
            self.measures = None

    print("Projects data")

    self.projects.info()
    if self.language == 'ptg':
        print("\n Co-h metrix file data")
        self.measures.info()
    print()

  def classification(self):
      print("Classification task")
      con_publicaciones = (self.projects[self.projects['numero_publicaciones'] > 0]).shape[0]  # 514-559
      sin_publicaciones = (self.projects[self.projects['numero_publicaciones'] <= 0]).shape[0]  # 525-615
      print('Publicaciones:', con_publicaciones, sin_publicaciones)
      number_of_samples = min(con_publicaciones, sin_publicaciones)
      print('Samples:', number_of_samples)
      print()
      iterations = 10
      #classifiers = ['DT', 'KNN', 'NB', 'SVM', 'RF']
      classifiers = ['MLP']
      methods = ['precision', 'recall', 'f1', 'accuracy']
      methods2 = ['pre_1', 'rec_1', 'pre_0', 'rec_0', 'accuracy']

      #container = {i:[[] for _ in range(4)] for i in classifiers}
      container = {i: [[] for _ in range(5)] for i in classifiers}
      print('Init iterations')
      print(container)


      for it in range(iterations):
          print('Iteration ' + str(it + 1))
          balanced_df = self.projects.groupby('label', as_index=False, group_keys=False).apply(lambda s: s.sample(number_of_samples, replace=True))
          obj = fExtractor.FeatureExtractor(language=self.language,method=self.metodo, text_type=self.text_type, corpus=balanced_df,
                             co_measures=self.measures, network_features=self.network_features,
                             word_embeddings=self.word_embeddings)
          features = obj.extract_features()
          print('Feature size: ', features.shape)
          Y = balanced_df["label"]


          #obj = Classification.Classification(features, Y)
          scaler = StandardScaler()  #### verificarrrrrrr
          features = scaler.fit_transform(features)

          obj = Classification.Classification(features, Y, per_class=True)
          scores = obj.get_scores()
          print(it + 1, scores)
          for c in scores:
              for i, score in enumerate(scores[c]):
                  container[c][i].append(score)
      print('End iterations')

      result_str = ''
      for c in container:
          print(c)
          for index, scores in enumerate(container[c]):
              avg = round(np.mean(scores),2)
              std = round(np.std(scores),2)
              #print(methods[index], ':' , avg, std)
              print(methods2[index], ':', avg, std)
              result_str += str(avg) + '(+/-' + str(std) + '),'
          print()
      result_str = result_str[:-1]
      print(result_str)


  def classification_test(self):
      con_publicaciones = (self.projects[self.projects['numero_publicaciones'] > 0]).shape[0]  # 514-559
      sin_publicaciones = (self.projects[self.projects['numero_publicaciones'] <= 0]).shape[0]  # 525-615
      print('Publicaciones:', con_publicaciones, sin_publicaciones)
      number_of_samples = min(con_publicaciones, sin_publicaciones)

      balanced_df = self.projects.groupby('label', as_index=False, group_keys=False).apply(lambda s: s.sample(number_of_samples, replace=True))

      obj = fExtractor.FeatureExtractor(language=self.language,method=self.metodo, text_type=self.text_type, corpus=balanced_df,
                                        co_measures=self.measures, network_features=self.network_features,
                                        word_embeddings=self.word_embeddings)

      features = obj.extract_features()
      print('Feature size: ', features.shape)
      Y = balanced_df["label"]

      #scaler = StandardScaler(with_mean=True, with_std=True)
      scaler = StandardScaler()
      features = scaler.fit_transform(features)

      #print(features)

      obj = Classification.Classification(features, Y, per_class=True)
      obj.classification2()
      #print(obj.get_scores())




  def classification_embeddings(self):
      print("Classification task")
      con_publicaciones = (self.projects[self.projects['numero_publicaciones'] > 0]).shape[0]  # 514-559
      sin_publicaciones = (self.projects[self.projects['numero_publicaciones'] <= 0]).shape[0]  # 525-615
      print('Publicaciones:', con_publicaciones, sin_publicaciones)
      number_of_samples = min(con_publicaciones, sin_publicaciones)
      print('Samples:', number_of_samples)
      print()
      iterations = 10
      #classifiers = ['DT', 'KNN', 'NB', 'SVM', 'RF']
      classifiers = ['MLP']
      methods = ['precision', 'recall', 'f1', 'accuracy']
      #container = {i: [[] for _ in range(4)] for i in classifiers}
      print('Init iterations')
      #print(container)
      limiars = [5,10,20,40,50]

      vector_limiars = []
      for i in range(len(limiars)+1):
          vector_limiars.append({i: [[] for _ in range(4)] for i in classifiers})

      for i in vector_limiars:
          print(i)

      for it in range(iterations):
          print('Iteration ' + str(it + 1))

          balanced_df = self.projects.groupby('label', as_index=False, group_keys=False).apply(lambda s: s.sample(number_of_samples, replace=True))
          obj = fExtractor.FeatureExtractor(method=self.metodo, text_type=self.text_type, corpus=balanced_df,
                                        co_measures=self.measures, network_features=self.network_features,
                                        word_embeddings=self.word_embeddings, limiars=limiars)

          limiar_features = obj.extract_features()
          Y = balanced_df["label"]

          limiar_scores = []
          for limiar in limiar_features:
              obj = Classification.Classification(limiar, Y)
              scores = obj.get_scores()
              limiar_scores.append(scores)
          print(it+1)

          for index, scores in enumerate(limiar_scores):
              print(index, scores)
              for c in scores:
                  for i, score in enumerate(scores[c]):
                      vector_limiars[index][c][i].append(score)
                      #container[c][i].append(score)
          print()

      print('End iterations')
      print()
      print('Haber final results\n')
      str_results = []
      for container in vector_limiars:
          print(container)
          result_str = ''
          for c in container:
              print(c)
              for index, scores in enumerate(container[c]):
                  avg = round(np.mean(scores), 2)
                  std = round(np.std(scores), 2)
                  print(methods[index], ':', avg, std)
                  result_str += str(avg) + '(+/-' + str(std) + '),'
          print()
          result_str = result_str[:-1]
          str_results.append(result_str)

      print('Final results ...')
      for result in str_results:
          print(result)


  def classification_embeddings_v2(self):
      print("Classification task")
      con_publicaciones = (self.projects[self.projects['numero_publicaciones'] > 0]).shape[0]  # 514-559
      sin_publicaciones = (self.projects[self.projects['numero_publicaciones'] <= 0]).shape[0]  # 525-615
      print('Publicaciones:', con_publicaciones, sin_publicaciones)
      number_of_samples = min(con_publicaciones, sin_publicaciones)
      print('Samples:', number_of_samples)
      print()

      #limiars = [5, 10, 20, 40, 50]
      limiars = [5, 10]

      balanced_df = self.projects.groupby('label', as_index=False, group_keys=False).apply(lambda s: s.sample(number_of_samples, replace=True))
      balanced_df.info()

      obj = fExtractor.FeatureExtractor(method=self.metodo, text_type=self.text_type, corpus=balanced_df,
                                        co_measures=self.measures, network_features=self.network_features,
                                        word_embeddings=self.word_embeddings, limiars=limiars)

      limiar_features = obj.extract_features()



