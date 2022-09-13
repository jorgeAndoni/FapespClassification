import pandas as pd
import datetime
from collections import Counter

def pruebas():
    #df = pd.read_csv('datasets/novas_features/d_medicina_results.csv') # 27 repts
    #df2 = pd.read_csv('datasets/novas_features/d_genetica_results.csv') # 30 repts
    #df3 = pd.read_csv('datasets/novas_features/d_quimica_results.csv') # 25 repts

    df = pd.read_csv('datasets/novas_features/r_medicina_results.csv') # 46 repts
    df2 = pd.read_csv('datasets/novas_features/r_odontologia_results.csv') #19 repts
    df3 = pd.read_csv('datasets/novas_features/r_veterinaria_results.csv') # 45 repts

    df = df.drop_duplicates()
    #df['AreaProject'] = ['medicina' for _ in range(df.shape[0])]

    df2 = df2.drop_duplicates()
    #df2['AreaProject'] = ['odontologia' for _ in range(df2.shape[0])]

    df3 = df3.drop_duplicates()
    #df3['AreaProject'] = ['veterinaria' for _ in range(df3.shape[0])]

    df.info()

    print(df.shape)
    print(df2.shape)
    print(df3.shape)

    df_pesqs = set(df['link_researcher'])
    df2_pesqs = set(df2['link_researcher'])
    df3_pesqs = set(df3['link_researcher'])
    print(len(df_pesqs), df_pesqs)
    print(len(df2_pesqs), df2_pesqs)
    print(len(df3_pesqs), df3_pesqs)
    print('Total:', len(df_pesqs)+len(df2_pesqs)+len(df3_pesqs))

    df_union = df_pesqs | df2_pesqs | df3_pesqs
    print('Union:', len(df_union), df_union)

    med_odon = df_pesqs & df2_pesqs
    med_vet = df_pesqs & df3_pesqs
    odon_vet = df2_pesqs & df3_pesqs
    print(len(med_odon), med_odon)
    print(len(med_vet), med_vet)
    print(len(odon_vet), odon_vet)

    df_all = pd.concat([df, df2, df3])
    print('nuevo shape all:',df_all.shape)
    pesquisadores_all = set(df_all['link_researcher'])
    print('pesquisadores:', len(pesquisadores_all), pesquisadores_all)
    print()
    df_all = df_all.drop_duplicates()
    print('nuevo shape sin duplicates:', df_all.shape)
    print('pesquisadores sin duplics:', len(pesquisadores_all), pesquisadores_all)

def join_datasets():
    df = pd.read_csv('datasets/novas_features/r_medicina_results.csv')  # 46 repts
    df2 = pd.read_csv('datasets/novas_features/r_odontologia_results.csv')  # 19 repts
    df3 = pd.read_csv('datasets/novas_features/r_veterinaria_results.csv')  # 45 repts
    df = df.drop_duplicates()
    df2 = df2.drop_duplicates()
    df3 = df3.drop_duplicates()
    df_all = pd.concat([df, df2, df3])
    df_all = df_all.drop_duplicates()
    df_all.to_csv('datasets/novas_features/research_projects_data.csv', index=False)

def parse_pt_date(date_string):
    MONTHS = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
              'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}
    FULL_MONTHS = {'janeiro': 1, 'fevereiro': 2, u'março': 3, 'abril': 4,
                   'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8,
                   'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12}

    '''Parses a date-time string and return datetime object
       The format is like this:
       'Seg, 21 Out 2013 22:14:36 -0200'
    '''
    if date_string is None:
        return ''

    date_info = date_string.lower().split()
    if date_info.count('de') == 2 or len(date_info) == 3:
        if ',' in date_info[0]:
            date_string = date_string.split(',')[1]
        date_info = date_string.lower().replace('de ', '').split()
        day, month_pt, year = date_info
        if len(month_pt) == 3:
            month = MONTHS[month_pt]
        else:
            month = FULL_MONTHS[month_pt]
        date_iso = '{}-{:02d}-{:02d}'.format(year, int(month), int(day))
        date_object = datetime.datetime.strptime(date_iso, '%Y-%m-%d')
        return date_object
    else:
        _, day, month_pt, year, hour_minute_second, offset = date_info

        if offset.lower() == 'gmt':
            offset = '+0000'
        offset_signal = int(offset[0] + '1')
        offset_hours = int(offset[1:3])
        offset_minutes = int(offset[3:5])
        total_offset_seconds = offset_signal * (offset_hours * 3600 +
                                                offset_minutes * 60)
        offset_in_days = total_offset_seconds / (3600.0 * 24)

        month = MONTHS[month_pt]
        datetime_iso = '{}-{:02d}-{:02d}T{}'.format(year, month, int(day),
                hour_minute_second)
        datetime_object = datetime.datetime.strptime(datetime_iso,
                '%Y-%m-%dT%H:%M:%S')
    return datetime_object - datetime.timedelta(offset_in_days)

def replace_area(list_area):
  nivel_1 = []
  nivel_2 = []
  nivel_3 = []
  for stri in list_area:
    stri = stri.split("-")
    if len(stri) >= 3:
      nivel_1.append(stri[0].strip())
      nivel_2.append(stri[1].strip())
      nivel_3.append(stri[2].strip())
    elif len(stri) == 2:
      nivel_1.append(stri[0].strip())
      nivel_2.append(stri[1].strip())
      nivel_3.append("")
    else:
      nivel_1.append(stri[0].strip())
      nivel_2.append("")
      nivel_3.append("")
  return nivel_1, nivel_2, nivel_3

def manages_datas():
    df = pd.read_csv('datasets/novas_features/research_projects_data.csv')
    '''
    df1 = pd.DataFrame(df.vigencia.str.split(' - ', 1).tolist(), columns=['start_vigencia', 'end_vigencia'])
    df = pd.concat([df1, df], axis=1)
    df['start_vigencia'] = df['start_vigencia'].apply(parse_pt_date)
    df['end_vigencia'] = df['end_vigencia'].apply(parse_pt_date)
    df.to_csv('datasets/novas_features/research_projects_data.csv', index=False)
    '''
    df['Area-Nivel1'], df['Area-Nivel2'], df['Area-Nivel3'] = replace_area(df["area"].tolist())
    df.info()
    df.to_csv('datasets/novas_features/research_projects_data.csv', index=False)




def process_dataset():
    df = pd.read_csv('datasets/novas_features/research_projects_data.csv')
    '''
    df1 = pd.DataFrame(df.vigencia.str.split(' - ', 1).tolist(), columns=['start_vigencia', 'end_vigencia'])
    df = pd.concat([df1, df], axis=1)
    df['start_vigencia'] = df['start_vigencia'].apply(parse_pt_date)
    df['end_vigencia'] = df['end_vigencia'].apply(parse_pt_date)
    df.to_csv('datasets/novas_features/research_projects_data.csv', index=False)
    '''

    fomentos = list(df['linha_fomento'])
    cuentas = Counter(fomentos)
    cuentas = {k: v for k, v in sorted(cuentas.items(), key=lambda item: item[1], reverse=True)}

    agrupados = dict()

    for index, fomento in enumerate(cuentas):
        print(index+1, fomento, cuentas[fomento])
        tipo = fomento[:fomento.find('-')]
        subtipo = fomento[fomento.find('-')+2:]
        count = cuentas[fomento]
        print(tipo)
        print(subtipo)
        print()
        if tipo in agrupados:
            agrupados[tipo].append((subtipo, count))
        else:
            agrupados[tipo] = [(subtipo, count)]
    print('\n')
    for grupos in agrupados:
        print(grupos, len(agrupados[grupos]))
        subs = agrupados[grupos]
        for i in subs:
            print('* ' + str(i))
        print()

def join_others(lista):
    counter = 0
    all_counter = 0
    nueva_lista = []
    result = dict()
    for pair in lista:
        if pair[0] == 'Outros':
            counter+=pair[1]
        else:
            #nueva_lista.append(pair)
            result[pair[0]] = pair[1]
        all_counter+=pair[1]
    if counter>0:
        #nueva_lista.append(('Outros', counter))
        result['Outros'] = counter
    return result, all_counter


if __name__ == '__main__':

    manages_datas()
    '''

    bolsa_values = ['Iniciação Científica', 'Mestrado', 'Doutorado',
                    'Doutorado Direto', 'Pós-Doutorado', 'Outros'
                    ]

    dict_fomentos = dict()
    dict_fomentos['Bolsas no Brasil '] = bolsa_values
    dict_fomentos['Bolsas no Exterior '] = bolsa_values
    #['Estágio de Pesquisa - Iniciação Científica', 'Estágio de Pesquisa - Mestrado',
    #'Estágio de Pesquisa - Doutorado', 'Estágio de Pesquisa - Doutorado Direto',
    #'Estágio de Pesquisa - Pós-Doutorado']
    dict_fomentos['Auxílio à Pesquisa '] = ['Regular', 'Outros']



    df = pd.read_csv('datasets/novas_features/research_projects_data.csv')
    pesquisador = '/pt/pesquisador/890/fernando-cendes/'
    #pesquisador = '/pt/pesquisador/43043/rodrigo-do-tocantins-calado-de-saloma-rodrigues/'
    #pesquisador = '/pt/pesquisador/3016/gabriel-forato-anhe/'
    #pesquisador = '/pt/pesquisador/690454/ivan-aprahamian/'
    #pesquisador = '/pt/pesquisador/682507/fabio-de-rezende-pinna/'

    #df_pesq = df[df['link_researcher']==pesquisador] # aqui tengo q filtrar el anio tambien
    df_pesq = df[(df['link_researcher'] == pesquisador) & (df['start_vigencia'] <= '2019-12-31')]
    df_pesq.info()

    fomentos = list(df_pesq['linha_fomento'])
    contagem = Counter(fomentos)
    organizado = dict()
    for fomento in contagem:
        count = contagem[fomento]
        tipo = fomento[:fomento.find('-')]
        subtipo = fomento[fomento.find('-')+2:]
        index_stagio = subtipo.find('Estágio de Pesquisa - ')
        if index_stagio != -1:
            subtipo = subtipo[subtipo.find('-') + 2:]
        if subtipo not in dict_fomentos[tipo]:
            subtipo = 'Outros'
        print(tipo, subtipo, count)

        if tipo in organizado:
            organizado[tipo].append((subtipo, count))
        else:
            organizado[tipo] = [(subtipo, count)]
        print()
    print('\n')

    type_fomentos = ['Auxílio à Pesquisa ', 'Bolsas no Brasil ', 'Bolsas no Exterior ']
    for i in type_fomentos:
        if i == 'Auxílio à Pesquisa ':
            columns = dict_fomentos[i]
        else:
            columns = bolsa_values
        print(i)
        if i in organizado:
            grupos = organizado[i]
            grupos, total = join_others(grupos)
            print(total, grupos)
            print(columns)
            vector = [grupos[x] if x in grupos else 0 for x in columns]
        else:
            vector = [0 for _ in range(len(columns))]
        print(vector)
        print()
    print('Total bolsas:',df_pesq.shape[0])

    '''









