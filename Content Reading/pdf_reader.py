import os
from pypdf import PdfReader

folder_path=r"C:\Users\aditya\Documents\aditya docs"
def get_pdf_files(folder_path):
    pdf_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def get_content_inside(file_path):
    reader=PdfReader(file_path)
    text=""
    for page in reader.pages:
        extracted=page.extract_text()
        #what if there is no extrated text what in that case- suggestion 1:see the file name instead to categorise
        if extracted:
            text+=extracted
    
    return text

def get_all_content(folder_path):
    pdf_files_paths=get_pdf_files(folder_path)
    checklist={}
    for file in pdf_files_paths:
        content=get_content_inside(file)
        checklist[os.path.basename(file)]=content

    return checklist

final_dict=get_all_content(folder_path)

print(final_dict)

