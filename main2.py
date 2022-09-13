from Manager2 import Manager

tp = ['d', 'r']
area_d = ['quimica', 'genetica', 'medicina']
area_r = ['medicina', 'odontologia', 'veterinaria']
metodo = 'co-metrix'
duracion = ['all-', 'all']

for pesq in tp:
    if pesq == 'd':
        selected_area = area_d
    else:
        selected_area = area_r
    for area in selected_area:
        for dur in duracion:
            print('-------------------------- Init testes -----------------------------')
            print(pesq, area, dur, metodo)
            obj = Manager(type_project=pesq, area=area, duracion=dur, metodo=metodo)
            obj.classification()
            print(pesq, area, dur, metodo)
            print('-------------------------- Fin testes -----------------------------')
            print('\n\n\n\n')