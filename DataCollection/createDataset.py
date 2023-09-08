import sys
import random
import csv

from reposClass import *


##find whether an element is new
def is_element_in_csv(element, csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        
        for row in reader:
            if element[0] in row[0]:
                return True
        return False


class createDataset:
    def createDataSet(csvFileName,trainingProbability,trainingReposFileName,testingReposFileName):

        with open(csvFileName,"r")as file:
            repositories = csv.reader(file)
            for repository in repositories:
                try:
                    ##check if we have seen exact snippet before
                    seenRepo=is_element_in_csv(repository,"permanentCsvFiles/Repos.csv")

                    ##skip if we have seen it before
                    if seenRepo == True:
                        print("not a new repository")
                        raise
                    with open("permanentCsvFiles/Repos.csv",'a',newline='') as allReposFile:
                        writer=csv.writer(allReposFile)
                        writer.writerow(repository)

                    ##find repository name 
                    val=repository[0]
                    print(val)
                    owner,repo,sha=readRepos.parse_api_url(val)

                    ###find if its a training repository:
                    trainingRepo=is_element_in_csv([repo],"permanentCsvFiles/trainingReposNames.csv")
                    testingRepo=is_element_in_csv([repo],"permanentCsvFiles/testingReposNames.csv")


                    if trainingRepo:###part of the training repos
                        with open (trainingReposFileName,'a', newline='') as trainingFile:
                            writer=csv.writer(trainingFile)
                            writer.writerow(repository)
                    
                    elif testingRepo:####part of the testing repos
                        with open (testingReposFileName,'a', newline='') as testingFile:
                            writer=csv.writer(testingFile)
                            writer.writerow(repository)
                    else :#### unique repo
                        if random.random() < trainingProbability:#### if training
                            with open (trainingReposFileName,'a', newline='') as trainingFile:
                                writer=csv.writer(trainingFile)
                                writer.writerow(repository)
                            with open("permanentCsvFiles/trainingReposNames.csv",'a', newline='') as trainingFileNames:
                                writer=csv.writer(trainingFileNames)
                                writer.writerow([repo])
                        else:## assigned testing
                            with open (testingReposFileName,'a', newline='') as testingFile:
                                writer=csv.writer(testingFile)
                                writer.writerow(repository)
                            with open("permanentCsvFiles/testingReposNames.csv",'a', newline='') as testingFileNames:
                                writer=csv.writer(testingFileNames)
                                writer.writerow([repo])
                except:
                    print("skipping "+repository[0])

                                
                

