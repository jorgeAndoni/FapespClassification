
from Manager2 import Manager
import sys

tp = 'd'
area = 'medicina'
metodo = 'network'
duracion = ['all-', 'all']
#duracion = ['all-']
#stops = [True, False]
stops = [True]

for d in duracion:
    for s in stops:
        print('-------------------------- Init testes -----------------------------')
        print(tp, area, metodo, d, s)
        obj = Manager(type_project=tp, area=area, duracion=d, metodo=metodo,remove_stops=s)
        obj.classification()
        print(tp, area, metodo, d, s)
        print('-------------------------- Fin testes -----------------------------')
        print('\n\n\n\n')

# medicina 487
# quimica
# genetica

'''
from Manager import Manager
obj = Manager(area='veterinaria', metodo='network', vigencia_23_24=True, network_features='local', use_embeddings=False)
#obj = Manager(language='eng',area='veterinaria', metodo='tfidf', vigencia_23_24=True)
#obj.classification()
#obj.classification_test()
#obj.classification_embeddings()
#obj.classification_embeddings_v2()
'''

'''
arguments = sys.argv
tp = arguments[1]
ar = arguments[2]
dur = arguments[3]
met = arguments[4]
rs = arguments[5]
if rs == 'yes':
    rs = True
else:
    rs = False

print(arguments)
#python3 main.py d medicina all- network yes
#obj = Manager(type_project='d', area='medicina', duracion='all-', metodo='network',remove_stops=True)
obj = Manager(type_project=tp, area=ar, duracion=dur, metodo=met,remove_stops=rs)
obj.classification()
'''
