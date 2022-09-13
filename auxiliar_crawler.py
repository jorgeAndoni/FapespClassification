import pandas as pd
from bs4 import BeautifulSoup, NavigableString
import urllib.request
import ssl
import re
import time
import csv

def load_project(area):
  path_projects = 'datasets/projects/'
  path = path_projects + area + '_18_24.csv'
  projects = pd.read_csv(path)
  projects = projects[(projects['end_vigencia'] <= '2015-12-31')]
  projects = projects.loc[projects["vigencia_months"] >= 23]
  return projects

class Crawler(object):

    def __init__(self, link):
        self.link = link
        self.soup = self.connect_to_fapesp(1)

    def connect_to_fapesp(self, attempts):
        try:
            link = 'https://bv.fapesp.br' + self.link
            print('Connecting to: ', link)
            context = ssl._create_unverified_context()
            wp = urllib.request.urlopen(link, context=context)
            page = wp.read()
            print('Connection OK :)')
            return BeautifulSoup(page, features="html.parser")
        except:
            if attempts == 5:
                print('Connection error with link:', self.link)
                return []
            else:
                attempts += 1
                print('Connection error, sleeping 25 seconds ...')
                time.sleep(20.0)  ############
                print('Attempting to connect ' + str(attempts))
                return self.connect_to_fapesp(attempts)

    def get_subjects(self):
        try:
            trs = self.soup.find_all('tr')
            assunto_keywords = ''
            assunto_ids = ''
            for tr in trs:
                tr_content = tr.text
                if 'Assunto(s)' in tr_content:
                    assunt_info = tr.find_all('td')[1]
                    keyword_data = assunt_info.find_all('a')
                    keywords = ''
                    keyword_ids = ''
                    for data in keyword_data:
                        link = data['href']
                        id = link[link.rfind('/') + 1:]
                        keywords += data.text + '##'
                        keyword_ids += id + '##'
                    assunto_keywords = keywords[:-2]
                    assunto_ids = keyword_ids[:-2]
            return [assunto_keywords, assunto_ids]
        except:
            return ['', '']

class SubjectCrawler(object):

    def __init__(self, df, path_file):
        self.df = df
        self.path = path_file
        self.init_csv_file()

    def init_csv_file(self):
        headers = ['link_researcher', 'assunto_keywords', 'assunto_ids']
        self.file_output = open(self.path, 'w')
        self.writer = csv.writer(self.file_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.writer.writerow(headers)
        self.file_output.flush()

    def crawling(self):
        links = list(self.df['link'])
        total = len(links)
        for index, link in enumerate(links):
            print(str(index + 1) + ' de ' + str(total) + ': ' + link)
            obj = Crawler(link)
            data = obj.get_subjects()
            data.insert(0, link)
            self.writer.writerow(data)
            self.file_output.flush()
            print(data)
            print('\n')

if __name__ == '__main__':

    medicina = load_project('medicina')
    odontologia = load_project('odontologia')
    veterinaria = load_project('veterinaria')
    #medicina = medicina.head(5)

    print('Crawling medicina ......')
    obj = SubjectCrawler(medicina, 'datasets/novo/medicina.csv')
    obj.crawling()

    print('Crawling odontologia ......')
    obj = SubjectCrawler(odontologia, 'datasets/novo/odontologia.csv')
    obj.crawling()

    print('Crawling veterinaria ......')
    obj = SubjectCrawler(veterinaria, 'datasets/novo/veterinaria.csv')
    obj.crawling()
