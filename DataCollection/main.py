from reposClass import *
from getUrls import *
from createDataset import *
from readURL import *
from findFunction import *
import os
import time

def num_elements_in_csv(file_name):
    num_elements = 0

    with open(file_name, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            num_elements += len(row)
    return num_elements

def last_folderNumber(path):
    
    folder_names = os.listdir(path)
    last_num = -1

    for folder_name in folder_names:
        if folder_name.startswith('snippet'):
            num_str = folder_name.replace('snippet', '')
            try:
                num = int(num_str)
                last_num = max(last_num, num)
            except ValueError:
                pass
    return last_num

def checkAndCreateDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")



###File Paths ######### ####just have to enter the root you want data to be stored in 
RootPath= r""
checkAndCreateDir(RootPath)
####Training Paths
TrainingPath=RootPath+'/training'
checkAndCreateDir(TrainingPath)
TrainingRawData=TrainingPath+'/database'
checkAndCreateDir(TrainingRawData)



TrainingFunctionedData=TrainingPath+'/functionedDatabase'
checkAndCreateDir(TrainingFunctionedData)
TrainingFunctionedDataPos=TrainingFunctionedData+'/pos'
checkAndCreateDir(TrainingFunctionedDataPos)
TrainingFunctionedDataNeg=TrainingFunctionedData+'/neg'
checkAndCreateDir(TrainingFunctionedDataNeg)
#####Testing Paths ############
TestingPath=RootPath+'/testing'
checkAndCreateDir(TestingPath)

TestingRawData=TestingPath+'/database'
checkAndCreateDir(TestingRawData)


TestingFunctionedData=TestingPath+'/functionedDatabase'
checkAndCreateDir(TestingFunctionedData)
TestingFunctionedDataPos=TestingFunctionedData+'/pos'
checkAndCreateDir(TestingFunctionedDataPos)
TestingFunctionedDataNeg=TestingFunctionedData+'/neg'
checkAndCreateDir(TestingFunctionedDataNeg)

################uncomment soooon ############################
github_API_URL='https://api.github.com/search/commits?q=null+pointer+dereference&page={i}&per_page=100&language=c'

firstPage=21200
while True:
    

    trainingProbability=.8
    # # ## Read all repositories into a csv file
    csvFileName,finalPage=get_repo_URLS.get_urls(firstPage,github_API_URL)


    csv_trainingRepositoryFileName="training"+str(firstPage)+"-"+str(finalPage)+".csv"
    
    csv_testingRepositoryFileName="testing"+str(firstPage)+"-"+str(finalPage)+".csv"

    with open (csv_trainingRepositoryFileName,'a', newline='') as trainingFile:
                            writer=csv.writer(trainingFile)
                            writer.writerow('')
    with open (csv_testingRepositoryFileName,'a', newline='') as testingFile:
                        writer=csv.writer(testingFile)
                        writer.writerow('')
                    
    # ####create Dataset of training and testing repositories
    createDataset.createDataSet(csvFileName,trainingProbability,csv_trainingRepositoryFileName,csv_testingRepositoryFileName)





    if not any(os.scandir(TrainingRawData)):
        trainingFileNum=0
    else :
        trainingFileNum=last_folderNumber(TrainingRawData)


    if not any(os.scandir(TestingRawData)):
        testingFileNum=0
    else :
        testingFileNum=last_folderNumber(TestingRawData)





    training_repos_num=num_elements_in_csv(csv_trainingRepositoryFileName)
    try:
        testing_repos_num=num_elements_in_csv(csv_testingRepositoryFileName)


        trainingSkips,savedTrainingRepos=read_repo_url.read_find_place(csv_trainingRepositoryFileName,TrainingRawData,trainingFileNum,training_repos_num)
        if trainingSkips<50 and os.path.exists(csv_trainingRepositoryFileName):
            os.remove(csv_trainingRepositoryFileName)

        testingSkips,savedTestingRepos=read_repo_url.read_find_place(csv_testingRepositoryFileName,TestingRawData,testingFileNum,testing_repos_num)
        if testingSkips and os.path.exists(csv_testingRepositoryFileName):
            os.remove(csv_testingRepositoryFileName)

        trainingStart=trainingFileNum
        trainingEnd=trainingFileNum+savedTrainingRepos

        testingStart=testingFileNum
        testingEnd=testingFileNum+savedTestingRepos


        functionedData.functionTheSnippet(TrainingRawData,TrainingFunctionedDataPos,TrainingFunctionedDataNeg,trainingStart,trainingEnd)


        functionedData.functionTheSnippet(TestingRawData,TestingFunctionedDataPos,TestingFunctionedDataNeg,testingStart,testingEnd)
        

    except:
        try:
            trainingSkips,savedTrainingRepos=read_repo_url.read_find_place(csv_trainingRepositoryFileName,TrainingRawData,trainingFileNum,training_repos_num)
            if trainingSkips<50 and os.path.exists(csv_trainingRepositoryFileName):
                os.remove(csv_trainingRepositoryFileName)
            trainingStart=trainingFileNum
            trainingEnd=trainingFileNum+savedTrainingRepos
            functionedData.functionTheSnippet(TrainingRawData,TrainingFunctionedDataPos,TrainingFunctionedDataNeg,trainingStart,trainingEnd)
        except:
            print("No data gathered")
    difference=finalPage-firstPage
    firstPage+=difference+1
    print("final Page: "+str(finalPage))
    try:
        os.remove(csv_testingRepositoryFileName)
        os.remove(csvFileName)
    except:
         print("Leaving files in env")
    
    time.sleep(60)




