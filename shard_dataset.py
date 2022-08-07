import os
import shutil
from os import listdir
from os.path import isfile, join
datadir = ''
for folder in  listdir(datadir):
    root = os.path.join(datadir, folder)
    onlyfiles = sorted([os.path.join(root, f) for f in listdir(root) if isfile(join(root, f))])
    for i in range(0,560):
        dest = os.path.join(root, f'{i:03d}')
        os.makedirs(dest, exist_ok=True)
        for j in range(i*100, i*100+100):
            img_path = os.path.join(dest, f'{j-i*100:08d}' + ".png")
            shutil.move(onlyfiles[j], img_path)