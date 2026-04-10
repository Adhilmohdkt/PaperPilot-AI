import os
import re

def rename_pdfs():
    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
    
    if not os.path.exists(data_folder):
        print(f"Data folder {data_folder} does not exist.")
        return

    renamed_count = 0
    pattern = re.compile(r"^\d{4}\.\d{5}(v\d+)?\.")

    for filename in os.listdir(data_folder):
        if filename.lower().endswith(".pdf"):
            # Remove ArXiv prefixes like "2601.16513v1."
            new_name = pattern.sub("", filename)
            # Remove underscores
            new_name = new_name.replace("_", " ")

            if new_name != filename:
                old_path = os.path.join(data_folder, filename)
                new_path = os.path.join(data_folder, new_name)
                
                os.rename(old_path, new_path)
                print(f"Renamed:\n  From: {filename}\n  To:   {new_name}\n")
                renamed_count += 1
                
    print(f"✅ Successfully renamed {renamed_count} files!")

if __name__ == "__main__":
    rename_pdfs()
