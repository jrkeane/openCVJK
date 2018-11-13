#!/usr/bin/python3
'''
Title: xml_find_repalce.py
Author: HR Hubbert
Created: 28 July 18
Last modified: 28 July 18
Version: 1.0
Version notes:
1.0     Given a filepath for saving and writing, find a string in an xml and replace
'''

import os
import numpy as np
# define path that we are looking at
# Ninox4704.xml
newPath = '/media/kookaburra/JimsDisk/Ninox/flight1images/newXML/'
labelIMGPath = '/media/kookaburra/JimsDisk/Ninox/flight1images/label/'
oldPath = '/media/kookaburra/JimsDisk/Ninox/flight1images/'
# toreplace = '/media/kookaburra/JimsDisk/Ninox/flight1images/'
filenames = []
imagenames = []
for file in os.listdir(labelIMGPath):
    if file.endswith(".xml"):
        filenames.append(file)
        file2 = (file[:-4]+'.jpg')
        imagenames.append(file2)

print(filenames)
print(len(filenames))
# for i in range(len(filenames)):
#     with open(oldpath+filenames[i], 'U') as f:
#         # Read in the file
#         filedata = f.read()
#         # Replace the target string
#     filedata = filedata.replace(toreplace, path)
#     filedata = filedata.replace('flight1images', 'pre_training_images')
# # Write the file out again
#     with open(path+filenames[i], "w") as f:
#         f.write(filedata)

# f.close()
