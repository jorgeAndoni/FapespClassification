from Manager4 import ProjectManager

area = 'medicina' #medicina odontologia veterinaria
iterations = 8
top_words = 20 #[20, 50]
#features = 'all' # all f1 f2 f3 f4 f5 f6 f7 f8 f9 f10
#features = ['f1', 'f2', 'f5', 'f7', 'f9', 'f10']
features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f13', 'f14', 'f15', 'f16']
#features = ['f1', 'f2', 'f5', 'f7', 'f9']
#feature = 'f17'
#file_name = area + '_' +str(iterations) + '_' + str(top_words) + '.txt'
#file_name = 'results/' + file_name
#print(file_name)
#out_file = open(file_name, 'w')

#self.classifiers = ['KNN', 'NB', 'SVM', 'RF', 'MLP']
obj = ProjectManager(area=area, iterations=iterations, top_words=top_words,
                     features=features, ensemble=False, feature_selection=100)
results = obj.classification()
#obj.feature_selection()
print('\n')



'''
for feature in features:
    print('Feature method:',feature)
    obj = ProjectManager(area=area, iterations=iterations, last_years=ly,
                         by_area=by_area, top_words=top_words, features=feature, ensemble=True)
    results = obj.classification()
    out_file.write(results + '\n')
    out_file.flush()
    print('\n')
print('Resultados finalizados')
'''

'''
features:
all (ya con features optimizadas)
f1 -> self.get_project_publications_features() #5  
f2 -> self.get_number_publications_citations() #4
f3 -> self.get_fomento_values() #8
f4 -> self.get_project_area_features() #3
f5 -> self.get_colaborator_features() #6
f6 -> self.get_project_vinculated_features() #4
f7 = self.get_subject_features()  # top-k
f8 = self.get_subject_features_v2() #top-k*2
f9 = self.university_features() # 2
f10 = self.university_exterior_features() #1
f11 =  get_subject_features_v3() #5
f12 = university_features_v2() ---
f13 = get_subject_features_a() #8
f14 = get_subject_features_b() # top-k*2
f15 = get_subject_features_c() #7
f16 = get_subject_features_d() #top-k*2
f17 = university_features_v3()  ----
'''



'''
Algunas paginas para hacer ensembles:
https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/
https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
https://towardsdatascience.com/ensemble-learning-using-scikit-learn-85c4531ff86a
https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
'''