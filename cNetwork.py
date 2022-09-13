
from nltk import bigrams
import igraph
from igraph import *
import numpy as np
from sklearn.metrics import pairwise_distances
import utils
from scipy import spatial
import xnet

class CNetwork(object):

  def __init__(self, document, word_embeddings=None, percentages=None, folder=None):
    self.document = document
    self.words = list(set(self.document))
    self.model = word_embeddings
    self.percentages = percentages
    self.vocab_index = {word: i for i, word in enumerate(self.words)}
    self.word_index = {index: word for index, word in enumerate(self.words)}
    self.folder = folder

  def create_network(self):
      edges = []
      string_bigrams = bigrams(self.document)
      for i in string_bigrams:
          edges.append((i[0], i[1]))
      network = Graph()
      network.add_vertices(self.words)
      network.add_edges(edges)
      network.simplify()
      print('Nodes:', len(self.words), '-', 'Edges:', len(network.get_edgelist()))
      return network

  def get_weights(self, edge_list):
      weights = []
      for edge in edge_list:
          word_1 = self.word_index[edge[0]]
          word_2 = self.word_index[edge[1]]
          v1 = self.model[word_1]
          v2 = self.model[word_2]
          w = 1 - spatial.distance.cosine(v1, v2)
          weights.append(w)
      return weights

  def create_network_janelas(self, window):
      matrix = np.zeros((len(self.words), len(self.words)))
      for index, word in enumerate(self.document):
          neighbors = utils.get_neighbors(self.document, index, window)
          word_index = self.vocab_index[word]
          for neighbor in neighbors:
              neighbor_index = self.vocab_index[neighbor]
              matrix[word_index][neighbor_index] = 1
      np.fill_diagonal(matrix, 0)
      network = igraph.Graph.Adjacency(matrix.tolist(), mode="undirected")
      print('Nodes:', len(self.words), '-', 'Edges:', len(network.get_edgelist()))
      weights = self.get_weights(network.get_edgelist())
      network.vs['name'] = self.words
      network.es['weight'] = weights
      return network

  def get_embedding_networks(self, network):
      networks = self.add_embeddings(network)
      networks.insert(0, network)
      prueba = [len(net.get_edgelist()) for net in networks]
      print('Num edges in networks:', prueba)
      return networks


  def add_embeddings(self, network):
      network_size = network.vcount()
      actual_edges = network.get_edgelist()
      num_edges = network.ecount()
      original_weight = network.es['weight']

      maximum_num_edges = int((network_size * (network_size - 1)) / 2)
      remaining_edges = maximum_num_edges - num_edges
      edges_to_add = []
      for percentage in self.percentages:
          value = int(num_edges * percentage / 100) + 1
          edges_to_add.append(value)
      print('Edges to add:',edges_to_add)
      matrix = []
      for word in self.words:
          embedding = self.model[word]
          matrix.append(embedding)

      matrix = np.array(matrix)
      similarity_matrix = 1 - pairwise_distances(matrix, metric='cosine')
      similarity_matrix[np.triu_indices(network_size)] = -1
      similarity_matrix[similarity_matrix == 1.0] = -1
      largest_indices = utils.get_largest_indices(similarity_matrix, maximum_num_edges)

      max_value = np.max(edges_to_add)
      counter = 0
      index = 0
      new_edges = []
      new_weights = []

      while counter < max_value:
          x = largest_indices[0][index]
          y = largest_indices[1][index]
          if not network.are_connected(x, y):
              new_edges.append((x, y))
              word_1 = self.word_index[x]
              word_2 = self.word_index[y]
              v1 = self.model[word_1]
              v2 = self.model[word_2]
              w = 1 - spatial.distance.cosine(v1, v2)
              new_weights.append(w)
              counter += 1
          index += 1

      networks = []
      for value in edges_to_add:
          edges = []
          weights = []
          edges.extend(actual_edges)
          weights.extend(original_weight)
          edges.extend(new_edges[0:value])
          weights.extend(new_weights[0:value])

          new_network = Graph()
          new_network.add_vertices(self.words)
          new_network.add_edges(edges)
          new_network.es['weight'] = weights
          networks.append(new_network)
      return networks


  def shortest_path(self, network, features):
      average_short_path = network.shortest_paths(features)
      result = []
      for path in average_short_path:
          average = float(np.divide(np.sum(path), (len(path) - 1)))
          result.append(average)
      return result

  def accessibility(self, network, features):
      in_network = self.folder + 'auxiliar_network.xnet'
      extra_file = self.folder + 'acc_results.txt'
      xnet.igraph2xnet(network, in_network)
      path_command = './CVAccessibility2 -l 1 ' +  in_network + ' > ' + extra_file
      os.system(path_command)
      accs_values = utils.read_result_file(extra_file)
      result = []
      for word in features:
          node = network.vs.find(name=word)
          result.append(accs_values[node.index])
      return result

  def eigenvector(self, all_eigen, features):
      indexes = [self.vocab_index[i] for i in features]
      values = [all_eigen[i] for i in indexes]
      return values

  def get_local_features(self, network, word_features):
      found_features = []
      for word in word_features:
          try:
              node = network.vs.find(name=word)
          except:
              node = None
          if node is not None:
              found_features.append(word)

      #medidas = ['dgr', 'pr', 'btw', 'cc', 'sp', 'eigen', 'clos', 'accs']
      dgr = network.degree(found_features)
      pr = network.pagerank(found_features)
      btw = network.betweenness(found_features)
      cc = network.transitivity_local_undirected(found_features)
      sp = self.shortest_path(network,found_features)
      all_eigen = network.eigenvector_centrality()
      eigen = self.eigenvector(all_eigen, found_features)
      clos = network.closeness(found_features)
      accs = self.accessibility(network, found_features)
      measures = [dgr, pr, btw, cc, sp, eigen, clos, accs]
      #print('Found features:', len(found_features) ,found_features)
      network_features = []
      for measure in measures:
          feature = [0.0 for _ in range(len(word_features))]
          for word, value in zip(found_features, measure):
              feature[word_features[word]] = value
          network_features.extend(feature)

      network_features = np.array(network_features)
      network_features[np.isnan(network_features)] = 0
      print('Len features:', network_features.shape)
      return network_features