import pandas as pd
from collections import Counter
import numpy as np
from nltk import word_tokenize
import utils
from sklearn.feature_extraction.text import TfidfVectorizer
import igraph
from  scipy.spatial.distance import pdist

class ResearcherManager(object):

    def __init__(self, area='medicina', top_word=50, by_area=False,
                 last_years=False, flag_matrix=False):
        path = 'datasets/novas_features/'
        path2 = 'datasets/tf_idf_matrices/'
        self.top_words = top_word
        self.area = area
        self.by_area = by_area
        self.last_years = last_years


        self.df_researchers = pd.read_csv(path + 'research_researchers.csv')
        self.df_projects = pd.read_csv(path + 'research_projects_data.csv')
        self.df_publications = pd.read_csv(path + 'research_publications_updated.csv')
        self.df_publications_years = pd.read_csv(path + 'research_publication_years.csv')
        self.df_citations_years = pd.read_csv(path + 'research_citations_years.csv')
        self.target_projects = pd.read_csv('datasets/projects/' + area + '.csv')
        #self.target_projects = pd.read_csv('datasets/projects/'+ area + '_18_24.csv')
        #self.target_projects = self.target_projects[(self.target_projects['end_vigencia'] <= '2015-12-31')]
        #self.target_projects = self.target_projects.loc[self.target_projects["vigencia_months"] >= 23]
        #self.df_tfidf_matrix = pd.read_csv(path2+ area + '_' + str(top_word) + '.csv')
        if flag_matrix:
            self.df_tfidf_matrix = self.get_tfidf_table()
        else:
            self.df_tfidf_matrix = None


        self.csv_files = [self.df_researchers, self.df_projects, self.df_publications,
                          self.df_publications_years, self.df_citations_years, self.df_tfidf_matrix]

        #self.create_subject_network()

    def get_subject_vector(self, df):
        assunt_list = list(df['assunto_ids'].dropna())
        assuntos = []
        pruebita = ""
        for lista in assunt_list:
            lista = lista.split('##')
            assuntos.extend(lista)
            pruebita+=" ".join(lista) + " "
        return assuntos, pruebita

    def get_tfidf_table(self):
        area_dict = {'medicina': 'Medicina', 'odontologia': 'Odontologia',
                     'veterinaria': 'Medicina Veterinária'}
        area = area_dict[self.area]

        print('Creating tf-idf table')
        self.target_projects.info()
        print(self.target_projects.shape)

        print(list(self.target_projects.columns))
        r_links = list(self.target_projects['pesquisador_responsavel_link'])
        r_dates =  list(self.target_projects['start_vigencia'])
        r_projs = list(self.target_projects['link'])
        all_word_list = []
        corpus_subjects = []

        for index, (link, date) in enumerate(zip(r_links, r_dates)):
            if type(self.last_years) is int:
                year = int(date[0:4])
                last_n_date = str(year-self.last_years) + '-12-31'
                df = self.df_projects.loc[(self.df_projects['link_researcher'] == link) &
                                          (self.df_projects['start_vigencia'] <= date) &
                                          (self.df_projects['start_vigencia'] >= last_n_date)]
            else:
                df = self.df_projects.loc[(self.df_projects['link_researcher'] == link) &
                                          (self.df_projects['start_vigencia'] <= date)]

            if self.by_area:
                df = df.loc[df['Area-Nivel2']==area]

            #if self.by_area:
            #    df = self.df_projects.loc[(self.df_projects['link_researcher'] == link) &
            #                              (self.df_projects['start_vigencia'] <= date) &
            #                              (self.df_projects['Area-Nivel2'] == area)]
            #else:
            #    df = self.df_projects.loc[(self.df_projects['link_researcher'] == link) &
            #                              (self.df_projects['start_vigencia'] <= date)]

            vector_list, vector_str = self.get_subject_vector(df)
            all_word_list.extend(vector_list)
            corpus_subjects.append(vector_str)

        vect = TfidfVectorizer()
        tfidf_matrix = vect.fit_transform(corpus_subjects)

        df = pd.DataFrame(tfidf_matrix.toarray(), columns=vect.get_feature_names())

        all_word_list = Counter(all_word_list)
        top_k = all_word_list.most_common(self.top_words)
        top_words = [values[0] for values in top_k]
        print('Found top-words:',top_words)
        testdF = df.loc[:, top_words]
        testdF['pesquisador_responsavel_link'] = r_links
        testdF['link'] = r_projs
        print('\n')
        return testdF

    def create_subject_network(self):
        print('haber redecita?')
        self.target_projects.info()
        print(self.target_projects.shape)

        print(list(self.target_projects.columns))
        r_links = list(self.target_projects['pesquisador_responsavel_link'])
        r_dates = list(self.target_projects['start_vigencia'])
        r_projs = list(self.target_projects['link'])
        print('haber::',len(r_links), len(set(r_links)))
        all_assuntos = []
        for index, (link, date) in enumerate(zip(r_links, r_dates)):
            print(index+1, link, date)
            df = self.df_projects.loc[(self.df_projects['link_researcher'] == link) &
                                      (self.df_projects['start_vigencia'] <= date)]
            print(df.shape)
            assunt_list, _ = self.get_subject_vector(df)
            assunt_list = list(set(assunt_list))
            print(len(assunt_list), assunt_list)
            all_assuntos.append(assunt_list)




    def get_csv_files(self):
        return self.csv_files

    def get_researcher_features(self, link, year, area, last_years=False, link_proj='', proj_subjects='', selected_feats='all'):
        obj = ResearcherFeatures(self.csv_files, r_link=link, year=year, area=area,
                                 last_years=last_years, link_proj=link_proj, by_area=self.by_area,
                                 top_words=self.top_words, proj_subjects=proj_subjects)

        if type(selected_feats) is list:
            return obj.get_all_features_improved(selected_feats)

        if selected_feats == 'all':
            return obj.get_all_features()
        elif selected_feats == 'f1':
            return obj.get_project_publications_features()
        elif selected_feats == 'f2':
            return obj.get_number_publications_citations()
        elif selected_feats == 'f3':
            return obj.get_fomento_values()
        elif selected_feats == 'f4':
            return obj.get_project_area_features()
        elif selected_feats == 'f5':
            return obj.get_colaborator_features()
        elif selected_feats == 'f6':
            return obj.get_project_vinculated_features()
        elif selected_feats == 'f7':
            return obj.get_subject_features()
        elif selected_feats == 'f8':
            return obj.get_subject_features_v2()
        elif selected_feats == 'f9':
            return obj.university_features()
        elif selected_feats == 'f10':
            return obj.university_exterior_features()
        elif selected_feats == 'f11':
            return obj.get_subject_features_v3()
        elif selected_feats == 'f12':
            return obj.university_features_v2()
        elif selected_feats == 'f13':
            return obj.get_subject_features_a()
        elif selected_feats == 'f14':
            return obj.get_subject_features_b()
        elif selected_feats == 'f15':
            return obj.get_subject_features_c()
        elif selected_feats == 'f16':
            return obj.get_subject_features_d()
        elif selected_feats == 'f17':
            return obj.university_features_v3()
        else:
            return 'ERROR!'

class ResearcherFeatures(object):

    def __init__(self, csv_files, r_link, year=2020, area='medicina', last_years=False,
                 link_proj='', by_area=False, top_words=20, proj_subjects=''):
        self.researcher_link = r_link
        self.year = year
        self.df_researchers = csv_files[0]
        self.df_projects = csv_files[1]
        self.df_publications = csv_files[2]
        self.df_publications_years = csv_files[3]
        self.df_citations_years = csv_files[4]
        self.df_tf_idf_matrix = csv_files[5]
        self.date = str(year) + '-12-31'
        self.area = area
        self.last_n_years = last_years
        self.link_proj = link_proj
        self.by_area = by_area
        self.top_words = top_words
        self.project_subjects = proj_subjects

        if type(last_years) is int:
            self.last_n_date = str(year - last_years) + '-12-31'
            print('last n yearss:',self.last_n_date)
            self.pesq_projects = self.df_projects.loc[(self.df_projects['link_researcher'] == r_link) &
                                                  (self.df_projects['start_vigencia'] <= self.date) &
                                                      (self.df_projects['start_vigencia']>=self.last_n_date)]

            self.publications_data = self.df_publications.loc[(self.df_publications['Pesquisador_link'] == r_link) &
                                                              (self.df_publications['Year'] <= year) &
                                                              (self.df_publications['Year'] >= year-last_years)]
        else:
            self.last_n_date = None
            self.pesq_projects = self.df_projects.loc[(self.df_projects['link_researcher'] == r_link) &
                                                      (self.df_projects['start_vigencia'] <= self.date)]

            self.publications_data = self.df_publications.loc[(self.df_publications['Pesquisador_link'] == r_link) &
                                                              (self.df_publications['Year'] <= year)]

        print('Shape df pesq:',self.pesq_projects.shape)


    def get_number_publications(self, years):
        years = Counter(years.split('#'))
        counter = 0
        for year in years:
            if int(year)<=self.year: counter+=years[year]
        return counter

    def manage_publications(self, df):
        publications = list(df['PublicationYears'])
        numbers = []
        for index, years in enumerate(publications):
            if type(years) is not str: numbers.append(0)
            else:
                pubs = self.get_number_publications(years)
                numbers.append(pubs)
        return numbers

    def get_project_publications_features(self):
        vector_publications = self.manage_publications(self.pesq_projects)
        number_publications = sum(vector_publications)
        pubs_per_proj = np.mean(vector_publications)
        max_pubs_proj = max(vector_publications)
        projs_con_pubs = np.count_nonzero(vector_publications)
        taxa_pubs = (projs_con_pubs) / len(vector_publications)
        feature_vector = [number_publications, projs_con_pubs, max_pubs_proj, pubs_per_proj, taxa_pubs]
        print('Project pubs features:',feature_vector)
        return feature_vector

    def get_number_publications_citations(self):
        years = [str(x) for x in range(1990, self.year + 1)]
        pubs = self.df_publications_years[self.df_publications_years['Pesquisador_link'] == self.researcher_link]
        if pubs.shape[0] == 0:
            return 0,0,0,0
        pubs = list(pubs[years].iloc[0])
        cits = self.df_citations_years[self.df_citations_years['Pesquisador_link'] == self.researcher_link]
        cits = list(cits[years].iloc[0])
        if type(self.last_n_years) is int:
            index = self.last_n_years
            pubs = pubs[-index:]
            cits = cits[-index:]
        else:
            index = utils.get_index_first_publications(pubs)
            pubs = pubs[index:]
            cits = cits[index:]
        features = [sum(pubs), sum(cits), round(np.mean(pubs),2) , round(np.mean(cits),2)]
        print('Pub-cits features:', features)
        return features

    def get_fomento_values(self):
        bolsa_values = ['Iniciação Científica', 'Mestrado', 'Doutorado', 'Doutorado Direto', 'Pós-Doutorado', 'Outros']
        dict_fomentos = dict()
        dict_fomentos['Bolsas no Brasil '] = bolsa_values
        dict_fomentos['Bolsas no Exterior '] = bolsa_values
        dict_fomentos['Auxílio à Pesquisa '] = ['Regular', 'Outros']
        #dict_fomentos['Auxílio à Pesquisa '] = ['Regular']
        fomentos = list(self.pesq_projects['linha_fomento'])
        contagem = Counter(fomentos)
        organizado = dict()
        for fomento in contagem:
            count = contagem[fomento]
            tipo = fomento[:fomento.find('-')]
            subtipo = fomento[fomento.find('-') + 2:]
            index_stagio = subtipo.find('Estágio de Pesquisa - ')
            if index_stagio != -1: subtipo = subtipo[subtipo.find('-') + 2:]
            if subtipo not in dict_fomentos[tipo]: subtipo = 'Outros'
            if tipo in organizado: organizado[tipo].append((subtipo, count))
            else: organizado[tipo] = [(subtipo, count)]
        type_fomentos = ['Auxílio à Pesquisa ', 'Bolsas no Brasil ', 'Bolsas no Exterior ']
        final_dict = dict()
        for i in type_fomentos:
            if i == 'Auxílio à Pesquisa ': columns = dict_fomentos[i]
            else: columns = bolsa_values
            if i in organizado:
                grupos = organizado[i]
                grupos, total = utils.join_others(grupos)
                vector = [grupos[x] if x in grupos else 0 for x in columns]
            else:
                vector = [0 for _ in range(len(columns))]
            final_dict[i] = vector

        aux = final_dict['Auxílio à Pesquisa ']
        bb = final_dict['Bolsas no Brasil ']
        be = final_dict['Bolsas no Exterior ']
        joined = np.array([bb, be])
        feature_vector = []
        feature_vector.extend(aux)
        #feature_vector.extend(np.sum(aux))
        vector = np.sum(joined, axis=0).tolist()
        feature_vector.extend(vector)
        print('Fomentos:', feature_vector)
        return feature_vector

    def get_project_area_features(self):
        areas_1 = set(self.pesq_projects['Area-Nivel1'].dropna())
        areas_2 = set(self.pesq_projects['Area-Nivel2'].dropna())
        areas_3 = set(self.pesq_projects['Area-Nivel3'].dropna())
        feats = [len(areas_1), len(areas_2), len(areas_3)]
        print('Area features:', feats)
        return feats

    def get_colaborator_features(self):
        pesq_responsavel = set(self.pesq_projects['pesq_responsavel'].dropna())
        beneficiario = set(self.pesq_projects['beneficiario'].dropna())
        sup_exterior = set(self.pesq_projects['supervisor_exterior'].dropna())
        pesq_visitante = set(self.pesq_projects['pesq_visitante'].dropna())

        pesq_princ_count = list(self.pesq_projects['numero_pesq_principais'].dropna())
        pesq_assoc_count = list(self.pesq_projects['numero_pesq_associados'].dropna())
        number_projects_pesqPrinc = np.count_nonzero(pesq_princ_count)
        number_projects_pesqAssoc = np.count_nonzero(pesq_assoc_count)
        union_resp_benef = len(pesq_responsavel | beneficiario) - 1
        union_supExt_pesVis = len(sup_exterior | pesq_visitante)
        nro_authors = list(self.publications_data['Number_authors'])
        if len(nro_authors)!=0:
            mean_auts_per_paper = np.mean(nro_authors)
            authors = list(self.publications_data['Authors'].dropna())
            lista_authors = []
            for i in authors:
                lista = i.split('##')
                lista = [s.lower() for s in lista]
                lista_authors.extend(lista)
            lista_authors = set(lista_authors) ### mejorar para eliminar parecidos
            number_authors = len(lista_authors)-1
        else:
            mean_auts_per_paper = 0
            number_authors = 0

        features = [union_resp_benef, union_supExt_pesVis, number_projects_pesqPrinc,
                    number_projects_pesqAssoc, number_authors, mean_auts_per_paper]
        print('Colaborator feats:',features)
        return features

    def get_project_vinculated_features(self):
        df = self.pesq_projects[self.pesq_projects['vinculado_auxilio'].notna()]
        vinculos = Counter(list(df['vinculado_auxilio']))
        publications = self.manage_publications(df)
        total_pubs = np.sum(publications)
        projects_pubs = np.count_nonzero(publications)
        #print(total_pubs, projects_pubs)
        try:
            features = [len(vinculos), len(publications), total_pubs/len(publications), projects_pubs/len(publications)]
        except:
            features = [0.0, 0.0, 0.0, 0.0]
        print('Projetos vinculados:',features)
        return features

    def university_exterior_features(self):
        try:
            pesq_univ_ext = Counter(list(self.pesq_projects['instituicao_exterior'].dropna())).most_common(1)[0][0]
            if self.last_n_date is not None:
                df = self.df_projects.loc[(self.df_projects['start_vigencia'] <= self.date) &
                                          (self.df_projects['start_vigencia'] >= self.last_n_date) &
                                          (self.df_projects['instituicao_exterior'] == pesq_univ_ext)]
            else:
                df = self.df_projects.loc[(self.df_projects['start_vigencia'] <= self.date) &
                                          (self.df_projects['instituicao_exterior'] == pesq_univ_ext)]

            publications = self.manage_publications(df)
            all_projects = len(publications)
            total_pubs = np.sum(publications)
            features = [total_pubs/all_projects]
        except:
            features = [0.0]
        print('External university feats:', features)
        return features

    def get_u_features(self, university):
        #universities = utils.get_university_list()
        area_dict = {'medicina':'Medicina', 'odontologia':'Odontologia',
                     'veterinaria':'Medicina Veterinária'}
        area = area_dict[self.area]

        if self.last_n_date is not None:
            df = self.df_projects.loc[(self.df_projects['start_vigencia'] <= self.date) &
                                      (self.df_projects['start_vigencia'] >= self.last_n_date) &
                                      (self.df_projects['University'] == university)]
        else:
            df = self.df_projects.loc[(self.df_projects['start_vigencia']<= self.date) &
                                      (self.df_projects['University']== university)]

        publications = self.manage_publications(df)
        all_projects = len(publications)
        projects_with_pubs = np.count_nonzero(publications)
        total_pubs = np.sum(publications)
        #feats = [all_projects, projects_with_pubs, total_pubs]
        try:
            feats = [projects_with_pubs/all_projects, total_pubs/all_projects]
        except:
            feats = [0.0, 0.0]


        if self.by_area:
            if self.last_n_date is not None:
                df2 = self.df_projects.loc[(self.df_projects['start_vigencia'] <= self.date) &
                                           (self.df_projects['start_vigencia'] >= self.last_n_date) &
                                           (self.df_projects['University'] == university) &
                                           (self.df_projects['Area-Nivel2'] == area)]
            else:
                df2 = self.df_projects.loc[(self.df_projects['start_vigencia'] <= self.date) &
                                           (self.df_projects['University'] == university) &
                                           (self.df_projects['Area-Nivel2'] == area)]



            publications = self.manage_publications(df2)
            area_projects = len(publications)
            area_projects_with_pubs = np.count_nonzero(publications)
            area_total_pubs = np.sum(publications)
            #feats2 = [area_projects, area_projects_with_pubs, area_total_pubs]
            try:
                feats2 = [area_projects_with_pubs/area_projects,  area_total_pubs/area_projects]
            except:
                feats2 = [0.0, 0.0]

            features = []
            features.extend(feats)
            features.extend(feats2)
            return features

        return feats

    def university_features(self):
        pesq_univs = Counter(list(self.pesq_projects['University'].dropna())).most_common(1)############
        all_feats = []
        for data in pesq_univs:
            university = data[0]
            u_features = self.get_u_features(university)
            all_feats.append(u_features)
        all_feats = np.array(all_feats)
        all_avg = np.mean(all_feats, axis=0)
        #all_sum = np.sum(all_feats, axis=0)
        print('University feats avg:', all_avg)
        #print('final:', all_sum)
        return all_avg

    def university_features_v2(self):
        u_list = utils.get_university_list()
        pesq_univsity = Counter(list(self.pesq_projects['University'].dropna())).most_common(1)[0][0]
        print(pesq_univsity)
        feature_vector = [0 for _ in range(len(u_list)+1)]
        if pesq_univsity in u_list:
            index = u_list[pesq_univsity]
        else:
            index = -1
        feature_vector[index] = 1
        print(feature_vector)
        return feature_vector

    def university_features_v3(self):
        u_list = utils.get_university_list()
        pesq_univs = Counter(list(self.pesq_projects['University'].dropna())).most_common(1)
        university = pesq_univs[0][0]
        print(university)

        df = self.df_projects.loc[(self.df_projects['start_vigencia'] <= self.date) &
                                  (self.df_projects['University'] == university)]

        publications = self.manage_publications(df)
        all_projects = len(publications)
        projects_with_pubs = np.count_nonzero(publications)


        try:
            success = projects_with_pubs/all_projects
        except:
            success = 0.0
        #print(success)


        feature_vector = [0 for _ in range(len(u_list) + 1)]
        if university in u_list:
            index = u_list[university]
        else:
            index = -1
        feature_vector[index] = success
        print(feature_vector)

        #return [success]
        return feature_vector


    def get_subject_features(self):
        values = self.df_tf_idf_matrix.loc[(self.df_tf_idf_matrix['pesquisador_responsavel_link']==self.researcher_link) &
                                           (self.df_tf_idf_matrix['link']==self.link_proj)]
        values = list(values.iloc[0, :])
        values = values[:-2]
        print('Len Subject features', len(values))
        return values

    def get_assunto_success_dict(self, all=True):
        if all:
            df_all = self.df_projects.loc[(self.df_projects['start_vigencia'] <= self.date)]
        else:
            df_all = self.pesq_projects

        df_all = df_all[df_all['assunto_ids'].notna()]
        all_assuntos = list(df_all['assunto_ids'])
        publications = self.manage_publications(df_all)
        asunt_counter = dict()
        for index, (asuntos, pubs) in enumerate(zip(all_assuntos, publications)):
            asuntos = asuntos.split('##')
            if pubs > 0:
                count_p = 1
            else:
                count_p = 0
            for asunto in asuntos:
                if asunto in asunt_counter:
                    asunt_counter[asunto][0] += 1
                    asunt_counter[asunto][1] += pubs
                    asunt_counter[asunto][2] += count_p
                else:
                    asunt_counter[asunto] = [1, pubs, count_p]
        return asunt_counter

    def get_subject_features_v2(self):
        # se podria modificar incluindo las topwords y sus pesos
        size_feature = self.top_words #10 20
        df = self.pesq_projects[self.pesq_projects['assunto_ids'].notna()]
        lista_assuntos = list(df['assunto_ids'])
        lista_assuntos = [i.split('##') for i in lista_assuntos]
        lista_assuntos = [item for sublist in lista_assuntos for item in sublist]
        lista_assuntos = Counter(lista_assuntos).most_common(size_feature)
        subject_weights = self.get_assunto_success_dict()
        feats1 = []
        feats2 = []
        for data in lista_assuntos:
            assunto = data[0]
            values = subject_weights[assunto]
            taxa_1 = values[1]/values[0]
            taxa_2 = values[2] / values[0]
            feats1.append(taxa_1)
            feats2.append(taxa_2)

        size = size_feature - len(feats1)
        if size != 0:
            vec = [0.0 for _ in range(size)]
            feats1.extend(vec)
            feats2.extend(vec)

        features = []
        #features.extend(feats1)
        #features.extend(feats2)
        features.append(feats1)
        features.append(feats2)
        features = np.array(features)
        features = np.mean(features, axis=0)

        #print('palabras:',list(self.df_tf_idf_matrix.columns))

        print('Success subjects:', features)
        return features

    def get_subject_vector(self, df):
        assunt_list = list(df['assunto_ids'].dropna())
        assuntos = []
        for lista in assunt_list:
            lista = lista.split('##')
            assuntos.extend(lista)
        assuntos = list(set(assuntos))
        return assuntos



    def get_subject_features_v3(self):
        print('Testing subjects v3')
        df_all = self.df_projects.loc[(self.df_projects['start_vigencia'] <= self.date)]
        df_all = df_all[df_all['assunto_ids'].notna()]
        #df_all.info()
        #print(df_all.shape)
        #researchers = set(df_all['link_researcher'])
        #print('nro pesqs:', len(researchers))
        df_cito = df_all.groupby(['link_researcher'])
        all_assutos = []
        pesquisadores = []
        for index, (name, group) in enumerate(df_cito):
            #print(index+1, name)
            assuntos = self.get_subject_vector(group)
            #print(len(assuntos), assuntos)
            all_assutos.append(assuntos)
            pesquisadores.append(name)
            #print()

        print('Calculating matrix ...')
        dist = lambda p1, p2: len(set(p1) & set(p2)) / len(set(p1) | set(p2))
        matrix = np.asarray([[dist(p1, p2) for p2 in all_assutos] for p1 in all_assutos])
        np.fill_diagonal(matrix, 0)
        print('Matrix finished')

        network = igraph.Graph.Adjacency((matrix>0).tolist(), mode="undirected")
        network.vs['name'] = pesquisadores

        print('Number of network nodes:', network.vcount())
        print('Number of edges:', len(network.get_edgelist()))

        try:
            dg = network.degree([self.researcher_link])[0]
            btw = network.betweenness([self.researcher_link])[0]
            pr = network.pagerank([self.researcher_link])[0]
            closs = network.closeness([self.researcher_link])[0]
            cc = network.transitivity_local_undirected([self.researcher_link], mode='zero')[0]
            feature_vector = [dg, btw, pr, closs, cc]
        except:
            feature_vector = [0.0 for _ in range(5)]

        print(feature_vector)
        return feature_vector

    def get_subject_features_a(self):
        print('Subject feats (all): global(success)-local(success)')
        df = self.pesq_projects[self.pesq_projects['assunto_ids'].notna()]
        subjects = list(df['assunto_ids'])
        subjects = [i.split('##') for i in subjects]
        subjects = [item for sublist in subjects for item in sublist]
        subjects = Counter(subjects)

        success_vector_global = []
        success_vector_pesq = []
        subject_weights_global = self.get_assunto_success_dict()
        subject_weights_pesq = self.get_assunto_success_dict(all=False)

        for subject in subjects:
            if subject in subject_weights_global:
                values = subject_weights_global[subject]
                success = values[2] / values[0]
            else:
                success = 0.0
            success_vector_global.append(success)
            if subject in subject_weights_pesq:
                values = subject_weights_pesq[subject]
                success = values[2] / values[0]
            else:
                success = 0.0
            success_vector_pesq.append(success)

        def stats(v):
            return [max(v), np.mean(v), np.std(v), np.count_nonzero(v) / len(v)]

        feature_vector_global = stats(success_vector_global)
        feature_vector_pesq = stats(success_vector_pesq)
        features = []
        features.extend(feature_vector_global)
        features.extend(feature_vector_pesq)
        print(features)
        return features

    def get_subject_features_b(self):
        print('Subject feats (top-k): global(success)-local(success)')
        df = self.pesq_projects[self.pesq_projects['assunto_ids'].notna()]
        subjects = list(df['assunto_ids'])
        subjects = [i.split('##') for i in subjects]
        subjects = [item for sublist in subjects for item in sublist]
        size_feature = self.top_words #
        subjects = Counter(subjects).most_common(size_feature)
        success_vector_global = []
        success_vector_pesq = []
        subject_weights_global = self.get_assunto_success_dict()
        subject_weights_pesq = self.get_assunto_success_dict(all=False)

        for subject in subjects:
            subject = subject[0]
            if subject in subject_weights_global:
                values = subject_weights_global[subject]
                success = values[2] / values[0]
            else:
                success = 0.0
            success_vector_global.append(success)
            if subject in subject_weights_pesq:
                values = subject_weights_pesq[subject]
                success = values[2] / values[0]
            else:
                success = 0.0
            success_vector_pesq.append(success)

        size = size_feature - len(success_vector_global)
        if size != 0:
            vec = [0.0 for _ in range(size)]
            success_vector_global.extend(vec)
            success_vector_pesq.extend(vec)

        features = []
        features.extend(success_vector_global)
        features.extend(success_vector_pesq)
        print(features)
        return features


    def get_subject_features_c(self):
        print('Haber subjects:', 'all pesq features (Diego versao)')
        df = self.pesq_projects[self.pesq_projects['assunto_ids'].notna()]
        subjects = list(df['assunto_ids'])
        subjects = [i.split('##') for i in subjects]
        subjects = [item for sublist in subjects for item in sublist]
        subjects = Counter(subjects)
        subject_weights_global = self.get_assunto_success_dict()
        subject_weights_pesq = self.get_assunto_success_dict(all=False)

        counts_subjects_global = []
        counts_subjects_pesq = []

        for subject in subjects:
            if subject in subject_weights_global:
                values = subject_weights_global[subject]
                success = values[2] / values[0]
            else:
                success = 0.0
            counts_subjects_global.append(success)

            if subject in subject_weights_pesq:
                values = subject_weights_pesq[subject]
                count = values[0]
            else:
                count = 0.0
            counts_subjects_pesq.append(count)

        def stats(v):
            return [np.max(v), np.mean(v), np.std(v), np.count_nonzero(v)]

        def stats2(v):
            return [np.max(v), np.mean(v), np.std(v)]

        f1 = stats(counts_subjects_global)
        f2 = stats2(counts_subjects_pesq)
        features = []
        features.extend(f1)
        features.extend(f2)
        print(features)
        return features

    def get_subject_features_d(self):
        print('Haber subjects:', 'all pesq features (top-k) (Diego versao)')
        df = self.pesq_projects[self.pesq_projects['assunto_ids'].notna()]
        subjects = list(df['assunto_ids'])
        subjects = [i.split('##') for i in subjects]
        subjects = [item for sublist in subjects for item in sublist]
        size_feature = self.top_words
        subjects = Counter(subjects).most_common(size_feature)

        subject_weights_global = self.get_assunto_success_dict()
        subject_weights_pesq = self.get_assunto_success_dict(all=False)
        counts_subjects_global = []
        counts_subjects_pesq = []

        for subject in subjects:
            subject = subject[0]
            if subject in subject_weights_global:
                values = subject_weights_global[subject]
                success = values[2] / values[0]
            else:
                success = 0.0
            counts_subjects_global.append(success)

            if subject in subject_weights_pesq:
                values = subject_weights_pesq[subject]
                count = values[0]
            else:
                count = 0.0
            counts_subjects_pesq.append(count)

        size = size_feature - len(counts_subjects_global)
        if size != 0:
            vec = [0.0 for _ in range(size)]
            counts_subjects_global.extend(vec)
            counts_subjects_pesq.extend(vec)

        features = []
        features.extend(counts_subjects_global)
        features.extend(counts_subjects_pesq)
        print(features)
        return features



    def get_all_features_improved(self, selected_features):
        dict_feats = {'f1':self.get_project_publications_features, 'f2':self.get_number_publications_citations,
                      'f3':self.get_fomento_values, 'f4':self.get_project_area_features,
                      'f5':self.get_colaborator_features, 'f6':self.get_project_vinculated_features,
                      'f7': self.get_subject_features, 'f8':self.get_subject_features_v2,
                      'f9':self.university_features, 'f10':self.university_exterior_features,
                      'f11':self.get_subject_features_v3, 'f12':self.university_features_v2,
                      'f13':self.get_subject_features_a, 'f14':self.get_subject_features_b,
                      'f15':self.get_subject_features_c, 'f16':self.get_subject_features_d,
                      'f17':self.university_features_v3
                      }

        feature_vector = []
        feature_sizes = []
        for feature in selected_features:
            vector = dict_feats[feature]()
            feature_sizes.append(len(vector))
            feature_vector.extend(vector)
        print('Features:', len(feature_vector), feature_vector)
        print('Sizes:',feature_sizes)
        return feature_vector


    def get_all_features(self):
        f1 = self.get_project_publications_features()
        f2 = self.get_number_publications_citations()
        #f3 = self.get_fomento_values() # parece q empeora en las 3 areas
        #f4 = self.get_project_area_features()
        f5 = self.get_colaborator_features()
        #f6 = self.get_project_vinculated_features() # parece q empeora en las 3 areas
        f7 = self.get_subject_features() # con area o sin area?
        #f8 = self.get_subject_features_v2()
        f9 = self.university_features()
        f10 = self.university_exterior_features() # no aporto mucho, ni mejoro ni empeoro
        #f11 = self.get_subject_features_v3()
        #f12 = self.university_features_v2()
        #f13 = self.get_subject_features_v5()
        features = []
        features.extend(f1)
        features.extend(f2)
        #features.extend(f3) ---
        #features.extend(f4)
        features.extend(f5)
        #features.extend(f6) ----
        features.extend(f7)
        #features.extend(f8)
        features.extend(f9)
        features.extend(f10)
        #features.extend(f11)
        #features.extend(f12)
        #features.extend(f13)
        return features


if __name__ == '__main__':

    #pesquisador = '/pt/pesquisador/890/fernando-cendes/'
    #pesquisador = '/pt/pesquisador/8907/renata-de-almeida-coudry/'
    pesquisador = '/pt/pesquisador/8847/irina-kerkis/'
    link_proj = '/pt/auxilios/28220/correlacao-entre-a-presenca-de-splicing-aberrante-e-mutacoes-sinonimas-no-gene-tp53-em-adenocarcinom/'
    obj = ResearcherManager()
    # feats = obj.get_researcher_features(pesquisador, 2015)
    # print(len(feats),feats)

    csvs = obj.get_csv_files()
    #objTest = ResearcherFeatures(csvs, pesquisador, 2015, last_years=False, area='medicina', link_proj=link_proj)  # , area='veterinaria')
    subjects = '13714##5588##176349##151913##7064'
    objTest = ResearcherFeatures(csvs, pesquisador, 2015, last_years=False, area='medicina',link_proj=link_proj, proj_subjects=subjects)

    #objTest.get_project_publications_features()
    #objTest.get_number_publications_citations()
    #objTest.get_fomento_values()
    #objTest.get_project_area_features()
    #objTest.get_colaborator_features()
    #objTest.get_project_vinculated_features()
    #objTest.get_subject_features()
    #objTest.get_subject_features_v2()
    #objTest.get_tfidf_all_table()
    #objTest.university_features()
    #objTest.university_exterior_features()
    #objTest.get_subject_features_v3()
    print('Haberr',len(objTest.university_features_v2()))
    #objTest.get_subject_features_v4()
    #objTest.get_subject_features_d()

    #features = objTest.get_all_features()
    #print(len(features), features)




