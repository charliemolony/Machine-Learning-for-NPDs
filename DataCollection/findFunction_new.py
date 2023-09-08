import os
from os import path


def find_function_by_declaration(file_name, declaration):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        function = []
        in_function = False
        bracket_count = 0
        for line in lines:
            if declaration in line:
                in_function = True
                function.append(line)
            if in_function:
                function.append(line)
            if '{' in line:
                bracket_count += 1
            if '}' in line:
                bracket_count -= 1
                if bracket_count == 0 and in_function:
                    function.pop(0)
                    return ''.join(function)
        return None



def parse_functions(file_path):
    functions = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "@@ -" in line:
                
                function=line.split(' @@')[1].strip()
                functions.append(function)
    functions=list(set(functions))
    return functions

def write_to_file(file_name, data,folder_path):
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(data)

class functionedData:
    def functionTheSnippet(folderPath,posPath,negPath,start,end):
        lostRepos=0
        for i in range (start,end):
            i=str(i)
        
            folder_path=folderPath+"/snippet"+i
            if path.exists(folder_path):   
                try:  
                    diffPath =folder_path+"/diff"+i+".txt"
                    greenPath=folder_path+"/green"+i+".txt"
                    redPath=folder_path+"/red"+i+".txt"


                    functions=parse_functions(diffPath)
                    negData=""
                    posData=""
                    
                    for function in functions:
                        negData=negData+find_function_by_declaration(greenPath,function)
                        posData=posData+find_function_by_declaration(redPath,function)

    
                    write_to_file("pos"+i+".txt",posData,posPath)
                    write_to_file("neg"+i+".txt",negData,negPath)
                except:
                    lostRepos+=1
                    print("skipping"+i)


            
        print("Lost Repos "+str(lostRepos))







