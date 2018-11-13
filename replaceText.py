# replaceText.py

import os
import numpy as np

oldPathName = '/media/kookaburra/JimsDisk/Ninox/flight1images/' # currently running off a USB, but change it.
fileName = 'Ninox4704.xml' # images go into here
openFile = oldPathName + fileName

newFolder = 'newXML/'
newPathName = oldPathName+newFolder+fileName


# print(openFile)
# print(newPathName)

f = open(openFile, 'r')
string = f.readlines()
# path_string = f.readlines()[3]
# lines[1] = str(round(strength, 2)) + "\n"
 # print f.readlines()[1:15]
print(string[1])
print("")
string[1]="<folder>newFolderName</folder>"
print(string)
# string.replace("<path>","PATH")
# print(string)

ff = open(newPathName, 'w')
for line in string:
        ff.write(line)
ff.close()
