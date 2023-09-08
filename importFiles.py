import pandas as pd
import os
from os import path
import csv
import sqlite3
# from tokenise import processData


fileNums=[]
null_texts=[]
fixed_texts=[]
diff_texts=[]
allFiles=[]

for i in range(1,50):
     i=str(i)
     folder_path = 'C:/Users/charl/Desktop/Final year Project/GitHubApi/database/snippet'+i
     if path.exists(folder_path):
        try:
            nullFile=  folder_path+"/red"+i+".txt"
            null_text=open(nullFile).read()
            null_texts.append(null_text)

            
            fixedFile =folder_path+"/green"+i+".txt"
            fixed_text=open(fixedFile).read()
            fixed_texts.append(fixed_text)
            
            allFiles.append(nullFile)
            allFiles.append(fixedFile)
                        
            fileNums.append(i)
                       
        except:
            print("skipping"+i)


df = pd.DataFrame({
    "File Number": fileNums,
    "pos": null_texts,
    "neg": fixed_texts
})

            ###Combine all textfiles to one text file
# with open('combined_file.txt', 'w') as outfile:
#     for filename in allFiles:
#         with open(filename) as infile:
#             outfile.write(infile.read())






conn = sqlite3.connect('RawData.db')
# # Export the DataFrame to the SQLite database
df.to_sql('text_files', conn, if_exists='replace')

# # Close the connection to the database
conn.close()

df.to_csv("CSVDataFrame.csv",index=False)


