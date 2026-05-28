folder=None
from content_grouper import getans
from image_grouping import get_image_clusters
import os
import shutil
from llm_naming import get_foldername
folder_path = r"C:\Users\aditya\Desktop\filefolder"
# folder_path = r"C:\Users\aditya\Desktop\big papers"
ans=getans(folder_path)
imagans=get_image_clusters(
        image_dir                  = r"C:\Users\aditya\Desktop\filefolder",
        embedding_cache_path       = r"C:\Users\aditya\Downloads\embedding_cacheok.json",
        caption_cache_path         = r"C:\Users\aditya\Downloads\caption_cacheok.json",
        text_embedding_cache_path  = r"C:\Users\aditya\Downloads\text_embedding_cacheok.json",
        clustering_mode            = "general",
        noise_assignment           = "none",
        clip_weight                = 0.6,
        blip_weight                = 0.4,
    )
print(f"The ans from files is {ans}\n\n\n")
print(f"The ans from images is {imagans}\n\n\n")
# finalans=ans+getans
# print(finalans)


for i,j in ans.items():
    llm_input=j['llm_input']
    clustername=get_foldername(llm_input) if llm_input!="other" else llm_input
    print("before")
    print(clustername)
    clustername = clustername.strip()          # removes starting/ending spaces and \n
    clustername = clustername.replace("\n", "")
    clustername = clustername.replace("\r", "")
    print("after")
    print(clustername)
    clusterpath=os.path.join(folder_path,clustername)
    os.makedirs(clusterpath,exist_ok=True)
    for z in j["files"]:
        oldpath=os.path.join(folder_path,z)
        newpath=os.path.join(clusterpath,z)
        shutil.move(oldpath,newpath)

for i,j in imagans.items():
    llm_input=j['llm_input']
    clustername=get_foldername(llm_input) if llm_input!="other" else llm_input
    print(clustername)
    clustername = clustername.strip()          # removes starting/ending spaces and \n
    clustername = clustername.replace("\n", "")
    clustername = clustername.replace("\r", "")
    print(clustername)
    clusterpath=os.path.join(folder_path,clustername)
    os.makedirs(clusterpath,exist_ok=True)
    for z in j["files"]:
        oldpath=os.path.join(folder_path,z)
        newpath=os.path.join(clusterpath,z)
        shutil.move(oldpath,newpath)