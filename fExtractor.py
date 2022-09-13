
import utils
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from random import randrange
from sklearn.preprocessing import StandardScaler
from cNetwork import CNetwork


class FeatureExtractor(object):

  #def __init__(self, method='co-metrix', text_type='resumo', stops=False, corpus=None, co_measures=None,
  #               word_embeddings=None, limiars=None, word_features=None, extra_folder=None, pesqObj=None,
  #               area='medicina', last_years=False, r_features='all'):


  def __init__(self,method='co-metrix', text_type = 'resumo', stops=False, corpus=None, co_measures=None,
               word_embeddings=None, limiars=None, word_features=None, extra_folder= None, pesqObj=None,
               research_parameters=None):

      self.method = method
      self.corpus = corpus
      self.file_co_measures = co_measures
      self.word_embeddings = word_embeddings
      self.limiars = limiars
      self.remove_stops = stops
      self.number_word_features = word_features
      self.path_folder = extra_folder
      self.pesqFextExtractor = pesqObj
      #vect_parameters = [self.area, self.ly, self.r_features, self.top_words]
      self.area = research_parameters[0]
      self.last_years = research_parameters[1]
      self.r_features = research_parameters[2]
      #self.top_words = vect_parameters[3]
      #method=self.metodo, corpus=balanced_df, pesqObj=reManager, research_parameters=vect_parameters

      if text_type == 'resumo':
          #self.text_column = 'texts_with_stop_words'  # without stopwords?
          self.corpus['resumo'] = self.corpus['resumo'].fillna(" ")
          self.corpus[text_type] = self.corpus['title'] + " " + self.corpus['resumo']
          self.text_column = text_type
          self.corpus[text_type] = self.corpus[text_type].fillna(" ")
      #    self.text_column = 'resumo'
      elif text_type == 'title_assunto':
          self.corpus['assunto'] = self.corpus['assunto'].fillna(" ")
          self.corpus[text_type] = self.corpus['title'] + " " + self.corpus['assunto']
          self.text_column = text_type
          #self.corpus[text_type] = self.corpus[text_type].fillna(" ")
      else:  # text_type == 'title':  # assunto
          self.text_column = text_type
          self.corpus[text_type] = self.corpus[text_type].fillna(" ")

  def extract_features(self):
      if self.method == 'freqs':
          print("Bag of words model")
          return self.get_freqs()
      elif self.method == 'tfidf':
          print("TF-IDF model")
          return self.get_tfidf()
      elif self.method == 'co-metrix':
          print("Co-metrix measures")
          return self.get_co_metrix()
      elif self.method == 'network':
          print("Network measures :)")
          return self.get_network_measures()
      elif self.method == 'random':
          print("Testing random measures")
          return self.get_random()
      elif self.method == 'researcher':
          return self.get_researcher_features()
      else:
          return 'ERROR'

  def get_researcher_features(self):
      pesquisadores = list(self.corpus['pesquisador_responsavel_link'])
      vigencias = list(self.corpus['start_vigencia'])
      link_projs = list(self.corpus['link'])
      subjects_ids = list(self.corpus['assunto_ids'])

      print(len(pesquisadores), pesquisadores)
      print(len(vigencias), vigencias)
      print(len(link_projs), link_projs)
      size = len(pesquisadores)
      all_features = []
      for index, (pesq, vig, link, subjects) in enumerate(zip(pesquisadores, vigencias, link_projs, subjects_ids)):
          year = int(vig[0:4])
          print(str(index + 1) + ' de ' + str(size), pesq, year, subjects)
          features = self.pesqFextExtractor.get_researcher_features(link=pesq, year=year, area=self.area,
                                                                    last_years=self.last_years, link_proj=link,
                                                                    proj_subjects = subjects ,
                                                                    selected_feats=self.r_features)
          all_features.append(features)
          print(features)
          print('\n\n')
      all_features = np.array(all_features, dtype=object)  #### normalizar!!! ?
      scaler = StandardScaler()
      all_features = scaler.fit_transform(all_features)
      return all_features

  def get_freqs(self):
      counts = CountVectorizer(min_df=100, max_df=0.9, ngram_range=(1, 2))
      #counts = CountVectorizer(min_df=10, max_df=0.5, ngram_range=(1, 2))
      self.corpus['processed'] = self.corpus[self.text_column].apply(utils.text_processing, args=(self.remove_stops,))
      X_counts = counts.fit_transform(self.corpus['processed'])
      return X_counts.toarray()

  def get_tfidf(self):
      tfidf = TfidfVectorizer(min_df=10, max_df=0.5, ngram_range=(1, 2))
      self.corpus['processed'] = self.corpus[self.text_column].apply(utils.text_processing, args=(self.remove_stops,))
      X_tfidf = tfidf.fit_transform(self.corpus['processed'])
      return X_tfidf.toarray()

  def get_co_metrix(self):
      project_ids = list(self.corpus['PIndex'])
      features = []
      for index in project_ids:
          df = self.file_co_measures[self.file_co_measures['PIndex']==index]
          vector = list(df.iloc[0, :])
          vector = vector[1:]
          features.append(vector)
      features = np.array(features)
      nans = np.isnan(features)
      features[nans] = 0
      scaler = StandardScaler()
      features = scaler.fit_transform(features)
      return features

  def get_network_features_from_text(self, texto, word_features):
      print('\nAnalizing texto')
      print(texto)
      print(word_features)
      '''
        j = 1 -> [red_normal, red_5%, red_10%, red_20%, red_50%]
        j = 2 -> [red_normal, red_5%, red_10%, red_20%, red_50%]
        j = 3 -> [red_normal, red_5%, red_10%, red_20%, red_50%]
      '''
      obj = CNetwork(document=texto, word_embeddings=self.word_embeddings, percentages=self.limiars,
                     folder=self.path_folder)

      networks = [obj.create_network_janelas(window=w + 1) for w in range(3)]
      networks_with_embeddings = [obj.get_embedding_networks(network) for network in networks]
      network_features = []
      for nets_janela in networks_with_embeddings:
          for network in nets_janela:
              for feats in word_features:
                  values = obj.get_local_features(network, feats)
                  network_features.append(values)
      print('\n\n\n')
      return network_features

  def get_network_measures(self):
      self.corpus['processed'] = self.corpus[self.text_column].apply(utils.text_processing_v2, args=(self.remove_stops,))
      word_features = [utils.get_top_words(self.corpus['processed'], number) for number in self.number_word_features]
      lista = list(self.corpus['processed'])
      size = len(lista)
      size_container = 3 * (len(self.limiars)+1) * len(self.number_word_features)
      #size_container = 2 * (len(self.limiars) + 1) * len(self.number_word_features)
      vector_container = [[] for _ in range(size_container)]
      print(vector_container)
      print('sizecito:',size_container)
      for count, texto in enumerate(lista):
          print(str(count+1) + ' de ' + str(size))
          features = self.get_network_features_from_text(texto, word_features)
          for index, feats in enumerate(features):
              vector_container[index].append(feats)
      scaled_features = []
      for feature in vector_container:
          feature = np.array(feature)
          scaler = StandardScaler()
          feature= scaler.fit_transform(feature)
          scaled_features.append(feature)
      return scaled_features

  def get_random(self):
      all_features = []
      for text in self.corpus[self.text_column]:
          features = [randrange(100) for _ in range(200)]
          all_features.append(features)
      return np.array(all_features)