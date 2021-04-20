
"""
# ppt2txt.py
# script recursively extractis text from all pptx files
# in current directory - and saves to a text file 
""" 

# pip install python-pptx
import os, sys, glob
from pptx import Presentation
import glob

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
basedir = dir_path.split("/")[-1]
print(basedir)

ss=""
for myfile in glob.glob('**/*pptx', recursive=True):
    if "~$" in myfile:
        continue
    prs = Presentation(myfile)
    print(myfile)
    print("----------------------")
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                txt = shape.text
                txt_arr = txt.strip().split("\n")
                txt_arr = [x.strip() for x in txt_arr]
                txt_arr = [x.replace("\r"," ") for x in txt_arr]
                txt_arr = [myfile+" : " + x + "\n" for x in txt_arr]
                txt = "".join(txt_arr)
                ss += txt
                print(txt)

fname = basedir+".txt"
print("writing to file ", fname)
fh = open(fname, "w")
fh.write(ss)
fh.close()
