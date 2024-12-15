import os 
import tarfile

# A script to work in conjuction of the download.py file. THis one just extracts the files and puts them in a dir.
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir,"images")
os.makedirs(output_dir, exist_ok=True)

for file_name in os.listdir(base_dir):
    if file_name.endswith(".tar.gz"):
        file_path = os.path.join(base_dir, file_name)
        print(f"Extracting file: {file_name}")

        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=output_dir)

print("Finished extrating x-ray images from .tar.gz files")