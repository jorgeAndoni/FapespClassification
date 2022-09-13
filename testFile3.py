from Manager3 import Manager

tp = 'r'
area_r = 'veterinaria' # 'medicina' 'odontologia' 'veterinaria'
metodo = 'co-metrix'
#metodo = 'tfidf'
duracion = 'all-'
#text_types = ['title_assunto', 'title', 'assunto'] #'resumo' --> considerar version pequena
text_types = ['resumo']

str_results = []
path_res = 'new_results/all-_' + area_r + '_' + metodo + '.txt'
file_output = open(path_res, 'w')

for texto in text_types:
    print('-------------------------- Init testes -----------------------------')
    print(texto)
    obj = Manager(type_project=tp, area=area_r, duracion=duracion, metodo=metodo,
                  remove_stops=False, text_type=texto)
    results = obj.classification()
    str_results.append(results)
    file_output.write(results + '\n')
    file_output.flush()
    print('-------------------------- Fin testes -----------------------------')
    print('\n\n\n\n')

file_output.close()
print('Final results ...')
for i in str_results:
    print(i)