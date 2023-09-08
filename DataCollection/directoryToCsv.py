import os
import pandas as pd
from os import path
# Path to the training directory
dir = r""



# List of subdirectories in the training directory
subdirs = ["neg", "pos"]

# List to store the data
data = []
# text_set = set() # set to keep track of unique text elements
# num_duplicates=0
# Loop through each subdirectory
for subdir in subdirs:
    # Get the path to the subdirectory
    subdir_path = os.path.join(dir, subdir)
    # Loop through each file in the subdirectory
    for filename in os.listdir(subdir_path):
        # Get the path to the file
        filepath = os.path.join(subdir_path, filename)
        # Read the content of the file
        with open(filepath, encoding="utf8") as f:
            # Read the first line of the text file into a variable called "url"
            url = f.readline().strip()
            text = f.read().strip()
            #if text not in text_set:
            # Check if the text element is already in the set
            #text_set.add(text)
                # Add the data to the list and set
            data.append({"text": text, "label": 1 if subdir == "pos" else 0, "url": url})




df = pd.DataFrame(data)
print("length of dataset "+str(len(df)))
#df.to_csv('synthetic_data.csv', index=False)
