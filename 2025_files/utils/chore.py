import os
from tqdm import tqdm

def remove_unnecessary_counter(path):
    files = os.listdir(path)
    for file in tqdm(files):
        file_dir = os.path.join(path, file)
        if not os.path.isfile(file_dir):
            continue
        
        if '-1' in file:
            new_name = file[:file.index('-1')]
        else:
            new_name = file
        
        new_file_dir = os.path.join(path, f"{new_name}.csv")
        
        os.rename(file_dir, new_file_dir)

if __name__ == "__main__":
    path = "D:/home/BCML/IITP/data/16channel_Emotion/Polar/"
    remove_unnecessary_counter(path)