import requests
from pprint import pprint
from github import Github
import csv
import re
import os
import requests
import base64
import csv
from urllib import parse
import re
from urllib.request import urlopen
import os
import shutil


username = ''
token = ''

class GetRepos:
        
    def print_repo(repo):
        # repository full name
        print("Full name:", repo.full_name)
        print("Contents:")
        for content in repo.get_contents(""):
            print(content)


    # search repositories by name

    def get_repos(page,url):
        
        response = requests.get(url,auth=((username,token)))
        print("Status code: ", response.status_code)
        # In a variable, save the API response.
        response_dict = response.json()
        
        # find total number of repositories
        if response.status_code!=200:
            return response.status_code,0,[]

        # Find out more about the repositories.
        repos_dicts = response_dict['items']
        total_count= response_dict['total_count']
        

        return response.status_code,total_count,repos_dicts

    def write_repositories_to_csv(repositories,csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            
            for repository in repositories:
                authRepos = repository["author"]
                if authRepos!=None:
                    repositoryData=[repository['url']]
                    writer.writerow(repositoryData)
                
                


            
            
class readRepos:

    def read_row(filepath,i):
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            for index, row in enumerate(reader):
                if index == i:
                    return row
            return None
    
    def read_element(filepath,i,FileNum):
        row =readRepos.read_row(filepath,i) 
        url =row[0]
        print(url,i+FileNum)
        return url

    def parse_url(url):
        match = re.search(r'https://github.com/([^/]+)/([^/]+)/commit/([^/]+)', url)
        if match:
            owner = match.group(1)
            repo = match.group(2)
            sha = match.group(3)
            return owner, repo, sha

    
    def parse_api_url(url):
        match = re.search(r'https://api.github.com/repos/([^/]+)/([^/]+)/commits/([^/]+)', url)
        if match:
            owner=match.group(1)
            repo=match.group(2)
            sha=match.group(3)
            return owner,repo,sha
        

    def find_path (owner,repo,sha):
        commit_url = f'https://api.github.com/repos/{owner}/{repo}/commits/{sha}'
        response = requests.get(commit_url, auth=(username,token))
        if response.status_code != 200:
            raise ValueError(f'Error getting commit details: {response.json()["message"]}')
        commit_data = response.json()
        parent_sha = commit_data['parents'][0]['sha']
        file_paths = [file['filename'] for file in commit_data['files']]
        return parent_sha,file_paths
    
    def write_to_file(file_name, data,folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(data)

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
    
    def get_diff(owner,repo,neg_sha,pos_sha,i,folder_path,exit,FileNum):
        if exit != -1:
            i=FileNum+i
            i=str(i)
            diffFilePath="diff"+i+".txt"
            diff_url =f"https://github.com/{owner}/{repo}/compare/{pos_sha}...{neg_sha}.diff"
            with urlopen(diff_url) as r:
                content = r.read().decode('utf-8')
                content=diff_url+'\n'+content
                readRepos.write_to_file(diffFilePath,content,folder_path)

    def createFolder(i,FileNum,folderPath):
        i=i+FileNum
        directory="snippet" +str(i)
        path = os.path.join(folderPath, directory) 
        os.mkdir(path)
        return path

    def parse_repo_file(filePath,i,FileNum, folderPath):
        folderPath=readRepos.createFolder(i,FileNum,folderPath)
        URL=readRepos.read_element(filePath,i,FileNum)
    
        owner,repo,neg_sha = readRepos.parse_api_url(URL)
        pos_sha,path = readRepos.find_path(owner,repo,neg_sha)
        path = path[0]
        
        
        posURL = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={pos_sha}'
        negURL= f'https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={neg_sha}'
        

        #exit =readRepos.get_and_decode_snippet(negURL,"neg",i,folderPath,0,FileNum)
        exit =readRepos.get_and_decode_snippet(posURL,"pos",i,folderPath,0,FileNum)
        exit =readRepos.get_and_decode_snippet(negURL,"neg",i,folderPath,exit,FileNum)
        readRepos.get_diff(owner,repo,neg_sha,pos_sha,i,folderPath,exit,FileNum)

    


