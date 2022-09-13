
import urllib.request
import json
import pandas as pd
import re
import csv
import time

url = 'http://10.11.14.33/api/v1/metrix/all/m3tr1x01'

def request_co_metrix(texto):
    text = bytearray(texto, encoding='utf-8')
    req = urllib.request.Request(url, data=text, headers={'content-type': 'text/plain'})
    response = urllib.request.urlopen(req)
    y = json.loads(response.read().decode('utf8'))
    return y

def save_csv(file_path, container):
    #container.append((id, measures))
    columns = list(container[0][1].keys())
    header = list(columns)
    header.insert(0, 'PIndex')
    with open(file_path + '.csv', mode='w') as myFile:
        writer = csv.writer(myFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for content in container:
            id = content[0]
            row_dict = content[1]
            row = [row_dict[i] for i in columns]
            row.insert(0, id)
            writer.writerow(row)


def save_errors(path_error, error_files):
    columns = ['PIndex', 'resumo']
    with open(path_error + '.csv', mode='w') as myFile:
        writer = csv.writer(myFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns)
        for file in error_files:
            writer.writerow(file)


def get_coh_measures(ids, textos):
    total_texts = len(textos)
    container = []
    name_file = 'datasets/measures_doc/genetica/measures_part_'
    #name_file = 'projects_doctorado/measures/quimica/faltante_part_'
    error_files = []
    error_counting = 0
    #path_error = 'datasets/measures_doc/genetica/error_file'
    #path_error = 'projects_doctorado/measures/quimica/error_file'

    for index, (id, texto) in enumerate(zip(ids, textos)):
        #texto = re.sub(r"[-()\"#/@;:<>{}`+=~|!?]", "", texto)
        print(str(index + 1) + ' de ' + str(total_texts), id)
        try:
            measures = request_co_metrix(texto)
            #measures_co = pd.DataFrame(index=ids, columns=list(measures.keys()))
            #print(measures)
            container.append((id,measures))
        except:
            print('Error with project:', id)
            save_csv(name_file + str(error_counting), container)
            container = []
            error_counting+=1
            error_files.append([id, texto])
            print('Sleeping')
            time.sleep(30.0)

    if len(container)!=0:
        save_csv(name_file + str(error_counting), container)
    #save_errors(path_error, error_files)

def download_projects():
    #path = 'projects_doctorado/doctorado/quimica_all.csv'
    path = 'datasets/doctorado/quimica_all.csv'
    project = pd.read_csv(path)
    lista = list(project['resumo'])
    ids = list(project['PIndex'])

    lista = lista[0:2]
    ids = ids[0:2]
    #lista.append('')
    #ids.append(1200)
    get_coh_measures(ids, lista)
    print('Final :)')

def download_error_projects():
    #path = 'projects_doctorado/doctorado/quimica_all.csv'
    path = 'datasets/doctorado/quimica_all.csv'
    project = pd.read_csv(path)
    project.info()
    lista_ids = [40, 263, 641, 804]
    project = project[project['PIndex'].isin(lista_ids)]
    project.info()
    textos = project['resumo']
    lista_procesada = []
    for i in textos:
        i = i.encode('ascii', errors='ignore').decode()
        i = re.sub(r"[-()\"#/@;:<>{}`+=~|!?]", "", i)
        i = bytearray(i, encoding='utf-8')
        print(i)
        lista_procesada.append(i)
    #get_coh_measures(lista_ids, lista_procesada)





download_error_projects()
#UnicodeEncodeError: 'ascii' codec can't encode character '\xea' in position 42: ordinal not in range(128)