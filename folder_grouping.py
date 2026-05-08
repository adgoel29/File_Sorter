from content_grouper import getans
import os
import shutil
folder_path = r"C:\Users\aditya\Desktop\filefolder"
ans=getans(folder_path)
# print(ans)
for i,j in ans.items():
    clustername=os.path.join(folder_path,i)
    os.makedirs(clustername,exist_ok=True)
    print(clustername)
    for z in j:
        oldpath=os.path.join(folder_path,z)
        newpath=os.path.join(clustername,z)
        shutil.move(oldpath,newpath)
    # print(f"the cluster is {i}")
    # print(f"the files are {j}")
    # print("/n")

