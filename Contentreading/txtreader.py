import os

folder_path = r"C:\Users\aditya\Desktop\filefolder"

def get_txt_files(folder_path):
    txt_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files


def get_content_inside(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # fallback if encoding is different (common in Windows files)
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()


def get_all_content_txt(folder_path):
    txt_files_paths = get_txt_files(folder_path)
    checklist = {}
    
    for file in txt_files_paths:
        content = get_content_inside(file)
        checklist[os.path.basename(file)] = content

    return checklist


# final_dict = get_all_content(folder_path)
# print(final_dict)