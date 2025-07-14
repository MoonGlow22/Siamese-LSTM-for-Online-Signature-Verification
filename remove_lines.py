import os

"""
Removes lines containing "-100, -100", which indicate that the writing has stopped.
These lines are excluded to improve model training performance.

Note: This operation overwrites the original .txt files with the cleaned version.

"""

def clean_txt_files(main_folder):
    for root, _, files in os.walk(main_folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                new_lines = [line for line in lines if line.strip() != "-100, -100"]
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)

clean_txt_files(r"D:\yazılım\python\imza\Signatures")  # enter the folder path here

