import pandas as pd
from collections import Counter
import numpy as np
from nltk import word_tokenize
import utils

class ResearcherManager(object):

    def __init__(self, type_project='r'):
        path = 'datasets/novas_features/'
        if type_project == 'r':
            self.df_researchers = pd.read_csv(path + 'research_researchers.csv')
            self.df_projects = pd.read_csv(path + 'research_projects_data.csv')
            self.df_publications = pd.read_csv(path + 'research_publications_updated.csv')
            self.df_publications_years = pd.read_csv(path + 'research_publication_years.csv')
            self.df_citations_years = pd.read_csv(path + 'research_citations_years.csv')
        else:
            pass
        self.csv_files = [self.df_researchers, self.df_projects, self.df_publications,
                     self.df_publications_years, self.df_citations_years]

    def get_csv_files(self):
        return self.csv_files

    def get_researcher_features(self, link, year, area, fomento='bolsas_sum'):
        obj = ResearcherFeatures(self.csv_files, r_link=link, year=year, area=area, fomento=fomento)
        return obj.get_all_features()





class ResearcherFeatures(object):

    def __init__(self, csv_files, r_link, year=2020, fomento='bolsas_sum', area='medicina'):
        self.researcher_link = r_link
        self.year = year
        self.tipo_bolsa = fomento
        self.df_researchers = csv_files[0]
        self.df_projects = csv_files[1]
        self.df_publications = csv_files[2]
        self.df_publications_years = csv_files[3]
        self.df_citations_years = csv_files[4]

        self.date = str(year) + '-12-31'
        #self.pesq_projects = self.df_projects[(self.df_projects['link_researcher'] == r_link) & (self.df_projects['start_vigencia'] <= self.date)]
        self.pesq_projects = self.df_projects.loc[(self.df_projects['link_researcher'] == r_link) & (self.df_projects['start_vigencia'] <= self.date)]
        #df1 = df.loc[(df['a'] > 1)]
        self.researcher_data = self.df_researchers[self.df_researchers['Link']==r_link]
        self.publications_data = self.df_publications[(self.df_publications['Pesquisador_link']==r_link) & (self.df_publications['Year']<=year)]
        self.area = area

        lista = self.manage_publications()

        self.pesq_projects.loc[:, 'CountPubs'] = lista
        self.pesq_projects.loc[:, 'LabelPubs'] = [1 if x > 0 else 0 for x in lista]
        #self.pesq_projects.info()

        if area == 'medicina':
            self.max_subjects = 21
        elif area == 'odontologia':
            self.max_subjects = 24
        else:# veterinaria
            self.max_subjects = 38
        print('\n')

    def get_number_publications(self, years):
        years = Counter(years.split('#'))
        counter = 0
        for year in years:
            if int(year)<=self.year:
                counter+=years[year]
        return counter

    def manage_publications(self):
        publications = list(self.pesq_projects['PublicationYears'])
        numbers = []
        for index, years in enumerate(publications):
            if type(years) is not str:
                numbers.append(0)
            else:
                pubs = self.get_number_publications(years)
                print(index + 1, years, pubs)
                numbers.append(pubs)
        return numbers

    def get_project_publications_features(self):
        print('Testing projetct pubs features')
        print('haber')
        print(list(self.pesq_projects['CountPubs']))
        print(list(self.pesq_projects['LabelPubs']))

        lista_pubs = list(self.pesq_projects['CountPubs'])
        lista_pubs2 = list(self.pesq_projects['LabelPubs'])

        number_publications = sum(lista_pubs)
        pubs_per_proj = np.mean(lista_pubs)
        max_pubs_proj = max(lista_pubs)
        projs_con_pubs = sum(lista_pubs2)
        perc = (projs_con_pubs*100)/len(lista_pubs)

        print('Total pubs:', number_publications)
        print('Pubs media:', pubs_per_proj)
        print('Proj max pubs:', max_pubs_proj)
        print('Projects con pubs', projs_con_pubs, perc)




    def get_index_first_publications(self, lista):
        counter = 0
        for i in lista:
            if i==0:
                counter+=1
            else:
                return counter
        return -1

    def get_number_publications_citations(self):
        years = [str(x) for x in range(1990, self.year + 1)]
        #all_years = [str(x) for x in range(1990, 2021)]
        pubs = self.df_publications_years[self.df_publications_years['Pesquisador_link'] == self.researcher_link]
        if pubs.shape[0] == 0:
            return 0,0,0,0
        #print('haber',list(pubs[all_years].iloc[0]))
        pubs = list(pubs[years].iloc[0])
        cits = self.df_citations_years[self.df_citations_years['Pesquisador_link'] == self.researcher_link]
        cits = list(cits[years].iloc[0])
        index = self.get_index_first_publications(pubs)
        #print('Index:', index)
        pubs = pubs[index:]
        cits = cits[index:]
        #print(pubs)
        #print(cits)
        #print(np.mean(pubs), np.std(pubs), pubs)
        #print(np.mean(cits), np.std(cits), cits)
        return [sum(pubs), sum(cits), round(np.mean(pubs),2) , round(np.mean(cits),2)]

    def join_others(self,lista):
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

    def get_fomento_values(self):
        bolsa_values = ['Iniciação Científica', 'Mestrado', 'Doutorado', 'Doutorado Direto', 'Pós-Doutorado', 'Outros']
        dict_fomentos = dict()
        dict_fomentos['Bolsas no Brasil '] = bolsa_values
        dict_fomentos['Bolsas no Exterior '] = bolsa_values
        dict_fomentos['Auxílio à Pesquisa '] = ['Regular', 'Outros']
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
            if tipo in organizado:
                organizado[tipo].append((subtipo, count))
            else:
                organizado[tipo] = [(subtipo, count)]

        type_fomentos = ['Auxílio à Pesquisa ', 'Bolsas no Brasil ', 'Bolsas no Exterior ']
        final_dict = dict()
        for i in type_fomentos:
            if i == 'Auxílio à Pesquisa ': columns = dict_fomentos[i]
            else: columns = bolsa_values
            #print(i)
            if i in organizado:
                grupos = organizado[i]
                grupos, total = self.join_others(grupos)
                #print(total, grupos)
                vector = [grupos[x] if x in grupos else 0 for x in columns]
            else:
                vector = [0 for _ in range(len(columns))]
            #print(vector)
            final_dict[i] = vector
        print('Total bolsas:', self.pesq_projects.shape[0])
        for bolsa in final_dict:
            print(bolsa, final_dict[bolsa])
        '''
        bolsas_sum
        bolsas_union
        bolsas_br
        '''
        aux = final_dict['Auxílio à Pesquisa ']
        bb = final_dict['Bolsas no Brasil ']
        be = final_dict['Bolsas no Exterior ']
        joined = np.array([bb, be])
        feature_vector = []
        feature_vector.extend(aux)
        if self.tipo_bolsa == 'bolsas_sum': #bolsas_sum bolsas_union  bolsas_br
            vector = np.sum(joined, axis=0).tolist()
            feature_vector.extend(vector)
        elif self.tipo_bolsa == 'bolsas_union':
            feature_vector.extend(bb)
            feature_vector.extend(be)
        else: #bolsas_br
            feature_vector.extend(bb)
        return feature_vector


    def get_project_area_features(self):
        areas_1 = Counter(list(self.pesq_projects['Area-Nivel1'].dropna()))
        areas_2 = Counter(list(self.pesq_projects['Area-Nivel2'].dropna()))
        areas_3 = Counter(list(self.pesq_projects['Area-Nivel3'].dropna()))
        print(len(areas_1), areas_1)
        print(len(areas_2), areas_2)
        print(len(areas_3), areas_3)
        feats = [len(areas_1), len(areas_2), len(areas_3)]
        return feats

    def get_university_vector(self):
        universities = utils.get_university_list()
        vector = [0 for _ in range(len(universities)+1)]
        inst_sede = Counter(list(self.pesq_projects['University'].dropna()))
        for inst in inst_sede:
            if inst in universities:
                index = universities[inst]
            else:
                index = -1
            vector[index] = 1
        return vector

    def get_institute_features(self):
        feature_vector = self.get_university_vector()
        empresa = Counter(list(self.pesq_projects['empresa'].dropna()))
        if len(empresa)!=0:
            feature_vector.append(1)
        else:
            feature_vector.append(0)
        inst_visitante = Counter(list(self.pesq_projects['instituicao_visitante'].dropna()))
        if len(inst_visitante)!=0:
            feature_vector.append(1)
        else:
            feature_vector.append(0)
        return feature_vector

    def get_colaborator_features(self):
        pesq_responsavel = Counter(list(self.pesq_projects['pesq_responsavel'].dropna()))
        beneficiario = Counter(list(self.pesq_projects['beneficiario'].dropna()))
        sup_exterior = Counter(list(self.pesq_projects['supervisor_exterior'].dropna()))
        pesq_visitante = Counter(list(self.pesq_projects['pesq_visitante'].dropna()))

        pesq_princ_count = list(self.pesq_projects['numero_pesq_principais'].dropna())
        pesq_assoc_count = list(self.pesq_projects['numero_pesq_associados'].dropna())

        union_resp_benef = len(set(pesq_responsavel.keys())|set(beneficiario.keys()))-1
        union_supExt_pesVis = len(set(sup_exterior.keys())|set(pesq_visitante.keys()))

        print('Resp_bene', union_resp_benef)
        print('Ext-vist', union_supExt_pesVis)

        number_projects_pesqPrinc = np.count_nonzero(pesq_princ_count)
        number_projects_pesqAssoc = np.count_nonzero(pesq_assoc_count)

        #total_count_pesqPrinc = np.sum(pesq_princ_count) # nose si condierar
        #total_count_pesqAssoc = np.sum(pesq_assoc_count)

        print('Projects pesqPrinc:',number_projects_pesqPrinc)
        print('Projects pesqSec:',number_projects_pesqAssoc)

        nro_authors = list(self.publications_data['Number_authors'])  # Authors
        if len(nro_authors)!=0:
            mean_auts_per_paper = np.mean(nro_authors)
            authors = list(self.publications_data['Authors'].dropna())
            lista_authors = []
            for i in authors:
                lista = i.split('##')
                lista = [s.lower() for s in lista]
                lista_authors.extend(lista)
            lista_authors = Counter(lista_authors) ### mejorar para eliminar parecidos
            number_authors = len(lista_authors)-1
        else:
            mean_auts_per_paper = 0
            number_authors = 0
        print('Mean authors per paper:', mean_auts_per_paper)
        print('Authors:', number_authors)

        '''
        lista_authors = {k: v for k, v in sorted(lista_authors.items(), key=lambda item: item[1], reverse=True)}
        #for index, author in enumerate(lista_authors):
        #    print(index+1, author, lista_authors[author])
        print('Main pesquisador',self.researcher_data['Pesquisador'].iloc[0])
        '''
        feats = [union_resp_benef, union_supExt_pesVis, number_projects_pesqPrinc,
                 number_projects_pesqAssoc, mean_auts_per_paper, number_authors]
        return feats

    def get_projects_subjects(self):
        assunt_list = list(self.pesq_projects['assunto_keywords'].dropna())
        assuntos = []
        for lista in assunt_list:
            lista = lista.split('##')
            assuntos.extend(lista)
        assuntos = Counter(assuntos)
        assuntos = {k: v for k, v in sorted(assuntos.items(), key=lambda item: item[1], reverse=True)}
        return assuntos

    def get_all_subject_projects(self, by_area=False):
        print('date:',self.date)
        past_date = str(self.year-5) + '-12-31'
        if by_area:
            df = self.df_projects[(self.df_projects['start_vigencia'] <= self.date) &
                                  (self.df_projects['TargetArea']==self.area)]
        else:
            #df = self.df_projects[self.df_projects['start_vigencia'] <= self.date]
            df = self.df_projects[(self.df_projects['start_vigencia'] <= self.date) &
                                  (self.df_projects['start_vigencia'] >= past_date)]
        all_assunt_list = list(df['assunto_keywords'].dropna())
        all_assuntos = []
        for index, lista in enumerate(all_assunt_list):
            lista = lista.split('##')
            all_assuntos.extend(lista)
        all_assuntos = Counter(all_assuntos)
        max_value = all_assuntos.most_common(1)[0][1]
        assuntos_importance = {key:round(all_assuntos[key]/max_value,4) for key in all_assuntos}
        return assuntos_importance

    def get_subject_features(self):
        assuntos_importance = self.get_all_subject_projects(by_area=False)
        print('All assuntos: ',len(assuntos_importance))
        print('\n')

        assuntos = self.get_projects_subjects()
        print('Pesq assuntos:',len(assuntos))
        vec_importances = []
        for index, assunto in enumerate(assuntos):
            #assunt_project = assuntos[assunto]
            if assunto in assuntos_importance:
                assunt_all = assuntos_importance[assunto]
            else:
                assunt_all = 0.0
            vec_importances.append(assunt_all)
            #print(index+1, assunto, assunt_project, '->' ,assunt_all)
        print('Featuring')
        print(max(vec_importances), min(vec_importances), np.mean(vec_importances))
        #max_feat_size = self.max_subjects
        max_feat_size = 50
        vec_importances = vec_importances[0:max_feat_size]
        print(max(vec_importances), min(vec_importances), np.mean(vec_importances))
        size = max_feat_size - len(vec_importances)
        if size!=0:
            vec = [0.0 for _ in range(size)]
            vec_importances.extend(vec)

        return vec_importances

    def get_title_features(self):
        title = list(self.pesq_projects['title'].dropna())
        #self.pesq_projects.info()
        #print(title)
        all_words = []
        sizes = []
        for i in title:
            i = i.lower()
            words = word_tokenize(i)
            all_words.extend(words)
            sizes.append(len(words))
        all_words = Counter(all_words)
        print(len(all_words), all_words)
        print(max(sizes), min(sizes), np.mean(sizes))  # con 0 rows falla
        return [len(all_words), np.mean(sizes)]

    def get_project_vinculated_features(self):
        vinculos = Counter(list(self.pesq_projects['vinculado_auxilio'].dropna()))
        sumatoria = sum(list(vinculos.values()))
        print(len(vinculos), sumatoria)
        #for vinc in vinculos:
        #    print(vinc, vinculos[vinc])
        return [len(vinculos), sumatoria]

    def get_all_features(self):
        f1 = self.get_number_publications_citations()
        f2 = self.get_fomento_values()
        f3 = self.get_project_area_features()
        f4 = self.get_institute_features()
        f5 = self.get_colaborator_features()
        f6 = self.get_subject_features()
        f7 = self.get_title_features()
        f8 = self.get_project_vinculated_features()
        features = []
        features.extend(f1) #ok
        features.extend(f2) # ok
        features.extend(f3) # ok
        features.extend(f4) # ok
        features.extend(f5) #ok
        features.extend(f6) # ok
        features.extend(f7) # ok
        features.extend(f8)
        return features

def some_analysis():
    projects = pd.read_csv('datasets/projects/veterinaria_18_24.csv')
    projects = projects[(projects['end_vigencia'] <= '2015-12-31')]
    projects = projects.loc[projects["vigencia_months"] >= 23]
    projects.info()
    print()

    pesquisadores = list(projects['pesquisador_responsavel_link'])
    vigencias = list(projects['start_vigencia'])

    obj = ResearcherManager()
    csvs = obj.get_csv_files()
    sizes = []

    for index, (pesq, vig) in enumerate(zip(pesquisadores, vigencias)):
        year = int(vig[0:4])
        print(index+1, pesq, year)
        objTest = ResearcherFeatures(csvs, pesq, year)
        subjects = objTest.get_projects_subjects()
        sizes.append(len(subjects))
        print(len(subjects), subjects)
        print('\n')
    print('Haber de haberes')
    print(max(sizes), min(sizes), np.mean(sizes), np.std(sizes))




if __name__ == '__main__':

    #some_analysis()


    pesquisador = '/pt/pesquisador/890/fernando-cendes/'
    #pesquisador = '/pt/pesquisador/3312/jose-luiz-laus/'# 2012'
    #pesquisador = '/pt/pesquisador/7903/jacqueline-mendonca-lopes-de-faria/'
    #pesquisador = '/pt/pesquisador/177935/harley-francisco-de-oliveira/'

    obj = ResearcherManager()
    #feats = obj.get_researcher_features(pesquisador, 2015)
    #print(len(feats),feats)

    csvs = obj.get_csv_files()
    objTest = ResearcherFeatures(csvs, pesquisador, 2018)#, area='veterinaria')
    #print(objTest.get_institute_features())
    #feats = objTest.get_subject_features()
    objTest.get_project_publications_features()
    #print(len(feats), feats)




    #obj = ResearcherFeatures(r_link=pesquisador, year=2015) #bolsas_sum bolsas_union  bolsas_br
    #print(obj.get_number_publications_citations()) #ok
    #print(obj.get_fomento_values()) # okk
    #print(obj.get_project_area_features()) # ok
    #print(obj.get_institute_features()) # ok
    #print(obj.get_colaborator_features()) #casi ok, mejorar lo de duplicacion de autores
    #print(obj.get_subject_features()) # ok, but pesquisar
    #print(obj.get_title_features()) # ok, but pesquisar
    #print(obj.get_project_vinculated_features()) # ok
    #all_feats = obj.get_all_features()
    #print(len(all_feats), all_feats)

