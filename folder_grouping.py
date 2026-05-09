from content_grouper import getans
import os
import shutil
from llm_naming import get_foldername
folder_path = r"C:\Users\aditya\Desktop\filefolder"
ans=getans(folder_path)
# print(ans)
for i,j in ans.items():
    llm_input=j['llm_input']
    clustername=get_foldername(llm_input) if llm_input!="other" else llm_input
    print(clustername)
    clusterpath=os.path.join(folder_path,clustername)
    os.makedirs(clusterpath,exist_ok=True)
    for z in j["files"]:
        oldpath=os.path.join(folder_path,z)
        newpath=os.path.join(clusterpath,z)
        shutil.move(oldpath,newpath)
