import csv
import os.path

metric_type="FalseNegative"
# Specify the CSV file path
csv_file_path = metric_type+".csv"

# Specify the column name to loop through
column_name = "text"



save_path = r""+'/'+metric_type
# Open the CSV file for reading
with open(csv_file_path, encoding='utf8') as csv_file:
    # Create a CSV reader
    csv_reader = csv.DictReader(csv_file)
    i=0
    # Loop through each row in the CSV file
    for row in csv_reader:
        # Get the value of the specified column for the current row
        column_value = row[column_name]
        i+=1
        # Create a text file with the column value as the filename
        file_name = metric_type+str(i)+".c"
        completeName = os.path.join(save_path, file_name) 
        # f = open(completeName, "x")
        # Open the text file for writing
        with open(completeName, "w") as text_file:
            # Write the column value to the text file
            text_file.write(column_value)
            
        print(file_name)
