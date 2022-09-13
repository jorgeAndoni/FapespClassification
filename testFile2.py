from Manager3 import Manager

tp = ['r', 'd']
#area_d = ['quimica', 'genetica']#, 'medicina']
area_d = ['medicina']
area_r = ['veterinaria']
metodo = 'network'
duracion = 'all-'

for pesq in tp:
    if pesq == 'd':
        selected_area = area_d
    else:
        selected_area = area_r
    for area in selected_area:
        print('-------------------------- Init testes -----------------------------')
        print(pesq, area, duracion, metodo)
        obj = Manager(type_project=pesq, area=area, duracion=duracion, metodo=metodo, remove_stops=False)
        obj.classification()
        print(pesq, area, duracion, metodo)
        print('-------------------------- Fin testes -----------------------------')
        print('\n\n\n\n')