import csv
from collections import defaultdict

# specify the input CSV file and the column name
input_file = r""
text_column = 'text'

# create a dictionary to store the number of lines for each 'text' element
line_counts = defaultdict(int)

# open the input CSV file
with open(input_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    try:
        for row in reader:
            # count the number of lines in the 'text' element
            num_lines = row[text_column].count('\n') + 1
            # add the count to the dictionary
            line_counts[num_lines] += 1
    except:
        print("skipping")

# print the histogram
for num_lines, count in line_counts.items():
    print(f"{num_lines} lines: {count} occurrences")