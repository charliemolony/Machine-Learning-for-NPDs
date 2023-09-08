import sys
import random
import csv
from reposClass import *
from getUrls import *
from createDataset import *
from readURL import *
from findFunction import *

from reposClass import *
from createDataset import *


def get_and_decode_snippet(url,state,i,folder_path,exit,FileNum):
    if exit !=-1:
        response = requests.get(url,auth=(username,token))
        content = response.json()
        if response.status_code != 200:
            shutil.rmtree(folder_path)
            #raise ValueError(f'Error getting commit details: {response.json()["message"]}')
            return -1

        decode_content =  base64.b64decode(content['content']).decode()
        decode_content=url+'\n'+decode_content
        i=FileNum+i
        i=str(i)
        C_snippet = state +i +".txt"

        readRepos.write_to_file(C_snippet, decode_content,folder_path)
        return 0
    else: return -1

##find whether an element is new
def is_element_in_csv(element, csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        
        for row in reader:
            rowName=row[0]

            if element in rowName:
                return True
        return False



##training 50-

github_API_URL='https://api.github.com/search/commits?q=null+pointer+dereference&page={i}&per_page=100&language=c'
get_repo_URLS.get_urls(0,github_API_URL=github_API_URL)

        
