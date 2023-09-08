import csv
import os

def write_to_file(file_name, data,folder_path):
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(data)


states=['true_negative','true_positive','false_positive','false_negative']

for state in states:
    count = 0
    with open(state+'.csv','r') as stateFile:
        StateReader=csv.DictReader(stateFile)
        for stateRow in StateReader:
                
        # Open the CSV file for reading
            with open(r"", 'r') as Analysisfile:
                count+=1
                # Create a CSV reader object
                AnalysisReader = csv.DictReader(Analysisfile)

                # Loop over each row in the CSV file
                for AnalysisRow in AnalysisReader:
                    # Check if the 'url' column contains the desired string
                    if stateRow['url'] in AnalysisRow['url']:
                        # Print the corresponding 'text' column
                        content ='//'+str(stateRow['url'])+'\n'+AnalysisRow['text']
                        fileName=state+str(count)+'.c'
                        folderPath=r""
                        write_to_file(fileName,content,folderPath)


                        