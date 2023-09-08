import os
import csv

# Set the path of the directory containing the snippet directories
dir_path = r""

# Set the name of the output CSV file
csv_filename = "EntireFile.csv"

# Open the output CSV file in write mode
with open(csv_filename, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    # Write the header row of the CSV file
    writer.writerow(["text", "label","url"])

    # Loop through the snippet directories
    for subdir in os.listdir(dir_path):
        # Check if the item in the directory is a directory
        if os.path.isdir(os.path.join(dir_path, subdir)):
            # Loop through the files in the directory
            for file in os.listdir(os.path.join(dir_path, subdir)):
                # Check if the file is a text file and not named "diff.txt"
                if file.endswith(".txt") and not file.startswith("diff"):
                    # Get the label based on the filename
                    if file.startswith("pos"):
                        label = 1
                    elif file.startswith("neg"):
                        label = 0
                    else:
                        label = None

                    # If the label is not None, read the contents of the file and add it to the CSV file
                    if label is not None:
                        filename = os.path.join(subdir, file)
                        with open(os.path.join(dir_path, filename), encoding="utf8") as f:
                            url = f.readline().strip()
                            text = f.read().strip()
                        try:
                            writer.writerow([text, label,url])
                        except:
                            print("skipping")
