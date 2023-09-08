from reposClass import readRepos
import shutil
from os import path
import shutil
from os import path
import os
# Specify the 




class read_repo_url:
    def read_find_place(csvRepoName,folderPath,PreviousFileNum,number_of_files):
        numofSavedRepos=0
        numOfSkips=0
        for x in range(0,number_of_files): 
            try:
                readRepos.parse_repo_file(csvRepoName,x,PreviousFileNum,folderPath)
                numofSavedRepos+=1
            except:
                folder_Path = folderPath+r"\snippet"+str(x)
                if path.exists(folder_Path):
                    shutil.rmtree(folder_Path)
                numOfSkips+=1
                print("skipping"+str(x))
        return numOfSkips,numofSavedRepos

        

                

        






