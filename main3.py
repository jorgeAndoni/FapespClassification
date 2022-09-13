
from Manager2 import Manager

tp = ['d', 'r']
area_d = ['quimica', 'genetica', 'medicina']
area_r = ['medicina', 'odontologia', 'veterinaria']
metodo = 'freqs' # 'tfidf'
duracion = [ 'all', 'all-']
stops = [True, False]

str_results = []
path_res = 'results/all_' + metodo + '.txt'

for pesq in tp:
    if pesq == 'd':
        selected_area = area_d
    else:
        selected_area = area_r
    for area in selected_area:
        for dur in duracion:
            for st in stops:
                print('-------------------------- Init testes -----------------------------')
                print(pesq, area, dur, st, metodo)
                obj = Manager(type_project=pesq, area=area, duracion=dur, metodo=metodo, remove_stops=st)
                results = obj.classification()
                str_results.append(results)
                print(pesq, area, dur, st, metodo)
                print('-------------------------- Fin testes -----------------------------')
                print('\n\n\n\n')

print('Final results:')
file = open(path_res, 'w')
for line in str_results:
    print(line)
    file.write(line + '\n')
file.close()