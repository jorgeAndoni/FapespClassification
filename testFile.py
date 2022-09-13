from Manager3 import Manager

tp = ['r', 'd']
area_d = ['quimica', 'genetica', 'medicina']
area_r = ['medicina', 'odontologia', 'veterinaria']
#metodo = 'freqs'
metodo = 'tfidf'
duracion = 'all-'

str_results = []
path_res = 'new_results/all-_' + metodo + '.txt'
file_output = open(path_res, 'w')

for pesq in tp:
    if pesq == 'd':
        selected_area = area_d
    else:
        selected_area = area_r
    for area in selected_area:
        print('-------------------------- Init testes -----------------------------')
        print(pesq, area, duracion, metodo)
        obj = Manager(type_project=pesq, area=area, duracion=duracion, metodo=metodo, remove_stops=False)
        results = obj.classification()
        str_results.append(results)
        file_output.write(results + '\n')
        file_output.flush()
        print(pesq, area, duracion, metodo)
        print('-------------------------- Fin testes -----------------------------')
        print('\n\n\n\n')
file_output.close()

print('Final results ...')
for i in str_results:
    print(i)