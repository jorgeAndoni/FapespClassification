import pandas as pd
from gensim.models import Word2Vec
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import numpy as np
#import Orange
import matplotlib.pyplot as plt
import nltk
import os
import re


def get_neighbors(word_list, index, w):
  if index - w >= 0:
    left = word_list[index - w:index]
  else:
    left = word_list[:index]
  right = word_list[index + 1:index + 1 + w]
  return list(set(left + right))

def text_processing(text, stops=False):
  text = text.lower()
  for c in string.punctuation:
    text = text.replace(c, " ")
  #text = " ".join(text.split())
  text = ''.join([i for i in text if not i.isdigit()])
  text = word_tokenize(text)
  if stops:
    stopwords = nltk.corpus.stopwords.words('portuguese')
    stopwords = {i: x for x, i in enumerate(stopwords)}
    text = [word for word in text if word not in stopwords and len(word) > 2]
  text = " ".join(text)
  return text

def text_processing_v2(text, stops=False):
  text = text_processing(text, stops)
  text = word_tokenize(text)
  return text

def get_top_words(texts, n):
  all_words = []
  for text in texts:
    all_words.extend(list(set(text)))
  counts = Counter(all_words)
  features = counts.most_common(n)
  most_commom = dict()
  for index, feat in enumerate(features):
    most_commom[feat[0]] = index
  return most_commom

def get_w2v_vectors(paths, size=300):
  p1 = pd.read_csv(paths[0]+"projects/"+paths[1])
  p2 = pd.read_csv(paths[0]+"projects/"+paths[2])
  p3 = pd.read_csv(paths[0]+"projects/"+paths[3])
  p1['processed'] = p1['texts_with_stop_words'].apply(text_processing_v2)
  p2['processed'] = p2['texts_with_stop_words'].apply(text_processing_v2)
  p3['processed'] = p3['texts_with_stop_words'].apply(text_processing_v2)
  corpus = list(p1['processed'])
  corpus.extend(list(p2['processed']))
  corpus.extend(list(p3['processed']))

  model = Word2Vec(corpus, size=size, window=5, min_count=1, workers=4)
  words = model.wv.vocab
  model_dict = dict()
  for word in words:
    model_dict[word] = model.wv[word]
  return model_dict

def get_w2v_embeddings(size):
  p1 = pd.read_csv('datasets/projects/medicina_18_24.csv')
  p2 = pd.read_csv('datasets/projects/odontologia_18_24.csv')
  p3 = pd.read_csv('datasets/projects/veterinaria_18_24.csv')
  p4 = pd.read_csv('datasets/doctorado/genetica_all.csv')
  p5 = pd.read_csv('datasets/doctorado/medicina_all.csv')
  p6 = pd.read_csv('datasets/doctorado/quimica_all.csv')

  '''
  self.corpus[text_type] = self.corpus['title'] + " " + self.corpus['resumo']
  p1['resumo'] = p1['resumo'].fillna(" ")
  '''
  projects = [p1, p2, p3, p4, p5, p6]
  corpus = []
  for p in projects:
    p['resumo'] = p['resumo'].fillna(" ")
    p['resumo'] = p['title'] + " " + p['resumo']
    p['processed'] = p['resumo'].apply(text_processing_v2)
    corpus.extend(list(p['processed']))

  #p1['processed'] = p1['texts_with_stop_words'].apply(text_processing_v2)
  #p2['processed'] = p2['texts_with_stop_words'].apply(text_processing_v2)
  #p3['processed'] = p3['texts_with_stop_words'].apply(text_processing_v2)
  #p4['processed'] = p4['resumo'].apply(text_processing_v2)
  #p5['processed'] = p5['resumo'].apply(text_processing_v2)
  #p6['processed'] = p6['resumo'].apply(text_processing_v2)
  #corpus = list(p1['processed'])
  #corpus.extend(list(p2['processed']))
  #corpus.extend(list(p3['processed']))
  #corpus.extend(list(p4['processed']))
  #corpus.extend(list(p5['processed']))
  #corpus.extend(list(p6['processed']))
  model = Word2Vec(corpus, size=size, window=5, min_count=1, workers=4)
  words = model.wv.vocab
  model_dict = dict()
  for word in words:
    model_dict[word] = model.wv[word]
  return model_dict



def get_largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def special_process(content):
  sentences = sent_tokenize(content)
  content = ''
  for i in sentences:
    sentence = ' '.join(word_tokenize(i))
    sentence += '\n'
    content += sentence
  return content

def testing():
  path = 'rankings/veterinaria.csv'
  data = pd.read_csv(path)
  data = data.head(20)
  data.info()
  names = list(data['Feature'])
  avranks = list(data['Rank'])

  #names = ["first", "second", "third", "fourth", "quinto"]
  #avranks = [1.9, 3.2, 2.8, 3.3]
  #avranks = [2.3, 5.4, 6.9, 7.4, 8.1]

  cd = Orange.evaluation.compute_CD(avranks, 10)  # tested on 30 datasets
  Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=10, textspace=1.5)
  plt.show()


def manage(): #
  #path = 'datasets/projects/veterinaria_18_24.csv'
  path = 'datasets/doctorado/quimica_all.csv'
  projects = pd.read_csv(path)

  #df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
  projects = projects.rename(columns={"Title": "title", "Inicio": "start_vigencia",
                           "Fin":"end_vigencia", "Assunto":"assunto",
                           "Resumo":"resumo", "Publication":"label",
                            "Link": "link"})
  projects.info()
  #projects_18_24.loc[projects_18_24['numero_publicaciones'] <= 0, 'label'] = 0
  #projects_18_24.loc[projects_18_24['numero_publicaciones'] > 0, 'label'] = 1
  #print(projects_18_24['label'])
  #projects.to_csv('datasets/doctorado/quimica_all_.csv', index=False)

def csv_processing():
  #path = 'datasets/projects/veterinaria_18_24.csv'
  path = 'datasets/doctorado/medicina_all.csv'
  projects = pd.read_csv(path)
  projects.info()
  print(projects.columns)
  projects['PIndex'] = list(projects.index)
  projects.info()
  #projects.to_csv('datasets/doctorado/medicina_all_.csv', index=False)
  print()


def test():
  import pandas as pd
  from bs4 import BeautifulSoup, NavigableString
  import urllib.request
  import ssl
  import re
  link = 'https://www.urbandictionary.com/define.php?term=ACAB'
  context = ssl._create_unverified_context()
  wp = urllib.request.urlopen(link, context=context)
  page = wp.read()
  soup = BeautifulSoup(page, features="html.parser")
  print(soup.prettify())

def process_co_files():
  #path = 'datasets/measures_doc/quimica/'
  #path = 'datasets/measures_doc/medicina/'
  path = 'datasets/measures_doc/genetica/'
  project = pd.read_csv(path + 'measures_part_0.csv')# + str(i) + '.csv')
  print(project.shape)
  columns = list(project.columns)
  print(columns)
  project = project.reindex(sorted(project.columns), axis=1)
  columns = list(project.columns)
  print(columns)
  project.to_csv(path + 'genetica.csv', index=False)
  '''
  lista = os.listdir(path)
  print(lista)
  dfs = []
  for i in range(2):
    project = pd.read_csv(path + 'measures_part_' + str(i) + '.csv')
    print(project.shape)
    dfs.append(project)
  print('\n Final concat\n')
  projects = pd.concat(dfs)
  print(projects.shape)
  columns = list(projects.columns)
  print(columns)
  print('Haberrr')
  projects = projects.reindex(sorted(projects.columns), axis=1)
  columns = list(projects.columns)
  print(columns)
  print('\nNuevoo')
  nuevo = pd.read_csv(path + 'faltante_part_0.csv')
  print(nuevo.shape)
  columns = list(nuevo.columns)
  nuevo = nuevo.reindex(sorted(nuevo.columns), axis=1)
  print(columns)
  columns = list(nuevo.columns)
  print(columns)

  merged = pd.concat([projects, nuevo])
  print(merged.shape)
  merged.to_csv(path + 'final_medicina.csv', index=False)
  '''

def change(value):
  if value == 1:
    return 1.0
  else:
    return 0.0


def modify_projects():
  path = 'datasets/projects/veterinaria_18_24.csv'
  df = pd.read_csv(path)
  size = df.shape[0]
  df['PIndex'] = [i for i in range(size)]
  df.to_csv('datasets/projects/veterinaria_18_24_.csv',index=False)

  '''
  path = 'datasets/doctorado/quimica_all.csv'
  df = pd.read_csv(path)
  print(list(df['label']))
  df['label'] = df['label'].apply(change)
  print(list(df['label']))
  df.to_csv(path, index=False)
  '''

def read_result_file(path):
  index = 0
  result = dict()
  file = open(path)
  for line in file.readlines():
    line = line.rstrip('\n')
    value = float(line)
    result[index] = value
    index += 1
  return result

def join_avg_devitations(averages, deviations):
  results = []
  for a,d in zip(averages, deviations):
    value = str(a) + '(+/-' + str(d) + ')'
    results.append(value)
  return results


def verify_proyects():
  path = 'datasets/doctorado/medicina_all.csv'
  #path = 'datasets/projects/veterinaria_18_24.csv'
  df = pd.read_csv(path)
  print(df.shape)
  #df.info()
  resumos = list(df['resumo'])
  links =  list(df['link'])

  lista = ['/pt/bolsas/86795/prevalencia-de-auto-anticorpos-antiperoxidase-durante-a-gravidez-e-no-pos-parto/',
           '/pt/bolsas/88253/diagnostico-por-imagem-no-intersexo/']
  df = df[~df.link.isin(lista)]
  print(df.shape)
  #df.drop(['Cochice', 'Pima'])
  #new_df = sales[~sales.CustomerID.isin(badcu)]
  df.to_csv(path, index=False)

def some_stats():
  path = 'datasets/doctorado/medicina_all.csv'
  df = pd.read_csv(path)
  df = df.loc[(df['vigencia_months'] >= 36) & (df['vigencia_months'] <= 48)]
  print(df.shape)
  label= list(df['label'])
  print(Counter(label))



def probar():
  path = 'datasets/doctorado/medicina_all.csv'
  df = pd.read_csv(path)
  df.info()
  con_publicaciones = (df[df['label'] == 1.0]).shape[0]
  sin_publicaciones = (df[df['label'] == 0.0]).shape[0]
  print('Publicaciones:', con_publicaciones, sin_publicaciones)
  number_of_samples = min(con_publicaciones, sin_publicaciones)

  print()
  balanced_df = df.groupby('label', as_index=False, group_keys=False).apply(lambda s: s.sample(number_of_samples, replace=False))
  balanced_df.info()

  c1 = balanced_df[balanced_df['label']==1.0]
  c2 = balanced_df[balanced_df['label'] == 0.0]
  c1_link = list(c1['link'])
  c1_link = list(set(c1_link))

  c2_link = list(c2['link'])
  c2_link = list(set(c2_link))

  print('c1',c1.shape, len(c1_link))
  print('c2', c2.shape, len(c2_link))

def get_year_from_str(value):
  match = re.search('\d{4}', value)
  year = match.group(0) if match else '2020'
  return year

def similar(a, b):
  return round(SequenceMatcher(None, a, b).ratio(),4)

def search_similars(lista, threshold=0.6):
  values = dict()
  for actual in lista:
    #print(actual)
    auxiliar = list(lista)
    auxiliar.remove(actual)
    parecidos = []
    for word in auxiliar:
      similarity = similar(actual, word)
      if similarity>=threshold:
        #print(actual, '#', word, similarity)
        parecidos.append(word)
    values[actual] = parecidos
    #print()

  for word in values:
    print(word, values[word])

def modify_area(value):
  if value == 'Medicina':
    return 'medicina'
  if value == 'Medicina Veterinária':
    return 'veterinaria'
  if value == 'Odontologia':
    return 'odontologia'
  return ''

def process_area():
  df = pd.read_csv('datasets/novas_features/research_projects_data.csv')
  df.info()
  #areas = Counter(list(df['Area-Nivel2'].dropna()))
  areas = Counter(list(df['Area-Nivel2']))
  print(len(areas), areas)
  df['TargetArea'] = df['Area-Nivel2'].apply(modify_area)
  areas = Counter(list(df['TargetArea']))
  print(len(areas), areas)
  #df.to_csv('datasets/novas_features/research_projects_data.csv', index=False)

def get_university_list():
  universities = open('university_list.txt', 'r').readlines()
  universities = [u.rstrip('\n') for u in universities]
  universities = {u:index for index, u in enumerate(universities)}
  return universities

def convert_uni(inst):
  if type(inst) is not str:
    return ''
  find_index = inst.find('Universidade')
  if find_index != -1:
    university = inst[find_index:]
    university = university[:university.find('.')]
  else:
    find_index = inst.find('.')
    if find_index != -1:
      university = inst[:find_index]
    else:
      university = inst
  return university


def process_univ():
  df = pd.read_csv('datasets/novas_features/research_projects_data.csv')
  df.info()
  df['University'] = df['instituicao_sede'].apply(convert_uni)
  l1 = list(df['University'])
  l2 = list(df['instituicao_sede'])
  #df.to_csv('datasets/novas_features/research_projects_data.csv', index=False)


def find_year(str_value):
  find_index = str_value.find('Dissertação de Mestrado')
  find_index2 = str_value.find('Tese de Doutorado')
  if find_index!=-1 or find_index2!=-1:
    return ''
  match = re.match(r'.*([1-2][0-9]{3})', str_value)
  if match is not None:
    return match.group(1)
  return ''

def process_pubs_data(str_value):
  if type(str_value) is not str:
    return ''
  publications = str_value.split('###')
  years = ''
  for index, pub in enumerate(publications):
    year = find_year(pub)
    if len(year)!=0:
      years+=year+'#'
    #print(index+1, year, '->' , pub)
  years = years[:-1]
  return years

def proccess_tabelita():
  df = pd.read_csv('output_publications.csv')
  df.info()
  print()
  #pubs_data = list(df['StrPublications'])
  df['PublicationYears'] = df['StrPublications'].apply(process_pubs_data)
  df.info()
  df.to_csv('output_publications.csv', index=False)
  #for index, pubs in enumerate(pubs_data):
  #  years = process_pubs_data(pubs)
  #  print(index+1, years)


def get_index_first_publications(lista):
  counter = 0
  for i in lista:
    if i == 0:
      counter += 1
    else:
      return counter
  return -1


def join_others(lista):
  counter = 0
  all_counter = 0
  result = dict()
  for pair in lista:
    if pair[0] == 'Outros':
      counter += pair[1]
    else:
      result[pair[0]] = pair[1]
    all_counter += pair[1]
  if counter > 0:
    result['Outros'] = counter
  return result, all_counter

'''
    def pruebita(self):
        columns = list(self.projects.columns)
        print(columns)
        pesquisadores = list(self.projects['pesquisador_responsavel_link'])
        vigencias = list(self.projects['start_vigencia'])
        size = len(pesquisadores)
        anios = []
        #reManager = ResearcherManager(type_project=self.type_project)
        for index, (pesq, vig) in enumerate(zip(pesquisadores, vigencias)):
            year = int(vig[0:4])
            anios.append(year)
            print(str(index+1) + ' de ' + str(size),pesq, year)
            print('\n\n')
        counts = Counter(anios)
        years = [y for y in range(1990,2016)]
        vector = []
        for y in years:
            print(y, counts[y])
            vector.append(counts[y])
        print(max(anios), min(anios), np.mean(anios))

        plt.title('Projetos de ' + self.area)
        plt.xlabel('Ano do inicio do projeto')
        plt.ylabel('Numero de projetos')
        plt.plot(years, vector)
        plt.show()
'''

def get_acc(value):
  value = value[:value.find('(')]
  value = value.replace('.', ',')
  return value

def manage_data():
  path = 'tablita.txt'
  lines = open(path, 'r').readlines()
  for line in lines:
    line = line.rstrip('\n')
    values = line.split()
    values = [get_acc(v) for v in values]
    values = " ".join(values)
    print(values)
    #print()

def special_dist(x,y):
  return len(set(x)&set(y))/ len(set(x)|set(y))


def load_project(area):
  path_projects = 'datasets/projects/'
  path = path_projects + area + '_18_24.csv'
  projects = pd.read_csv(path)
  projects = projects[(projects['end_vigencia'] <= '2015-12-31')]
  projects = projects.loc[projects["vigencia_months"] >= 23]
  return projects

def some_counts():
  medicina = load_project('medicina')
  odontologia = load_project('odontologia')
  veterinaria = load_project('veterinaria')
  print(medicina.shape)
  print(odontologia.shape)
  print(veterinaria.shape)
  print(1602+890+979)

def get_subjects():
  area = 'veterinaria'

  medicina = load_project(area)
  medicina_assuntos = pd.read_csv('datasets/novo/' + area +'.csv')
  print(medicina.shape, medicina_assuntos.shape)
  #medicina_assuntos.info()

  #link = list(medicina['link'])
  #haber_assuntos = list(medicina['assunto'])
  assuntos = list(medicina_assuntos['assunto_keywords']) #assunto_ids
  assuntos_ids = list(medicina_assuntos['assunto_ids'])  # assunto_ids

  medicina['assunto_keywords'] = assuntos
  medicina['assunto_ids'] = assuntos_ids

  medicina.to_csv('datasets/novo/' + area +'_v2.csv')



if __name__ == '__main__':

  haber = [1,2,3]
  haber = 'ss'
  if type(haber) is list:
    print('Es lista')
  else:
    print('No es lista')








