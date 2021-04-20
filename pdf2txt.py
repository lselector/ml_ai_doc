
"""
# pdf2txt.py
# script recursively extracts text from all PDF files
# and saves to a text file 
# It looks either in current directory
# or in directories provided on command line (space-separaed)
# 
# This script uses binary "pdftotext"
# You can install it on Mac using brew:
#     brew install poppler
# This adds this executable:
#     /usr/local/bin/pdftotext@ -> ../Cellar/poppler/21.03.0_1/bin/pdftotext
# So now you can use it from cmd prompt:
#     pdftotext --help
#
# To use from Python, install wrapper:
#     pip install xpdf_python
#
# Then:
#     import xpdf_python
#     txt_arr = xpdf_python.to_text("myfile.pdf")
#     for txt in txt_arr:
#         print(txt)
#
""" 

import os, sys, glob
import xpdf_python

if len(sys.argv) <= 1:
    mypath = os.path.dirname(os.path.realpath(__file__))
    mydirs = [mypath]
    print("looking for pdf files in script directory: ", mypath)
else:
    mydirs = []
    for argpath in sys.argv[1:]:
        if not os.path.isdir(argpath):
            print(f"{argpath} is not a directory, exiting ...")
            sys.exit()
        mypath = os.path.realpath(argpath)
        print(argpath, "   =>  ", mypath)
        mydirs += [mypath]

for dir_path in mydirs:
    print(dir_path)
    basedir = dir_path.split("/")[-1]
    fname_out = dir_path + "/" + basedir+"_pdf.txt"

    counter = 0
    myfiles = glob.glob(dir_path+'/**/*pdf', recursive=True)
    myfiles += glob.glob(dir_path+'/**/*PDF', recursive=True)
    ss = ""
    for myfile in sorted(myfiles):
        if "~$" in myfile:
            continue
        if "OLD/" in myfile:
            continue
        counter += 1
        if counter > 3:
            pass # break
        print("-"*20, myfile)
        myfile_short = myfile.split(dir_path)[1][1:]
        # print(myfile_short,"\n")
        # print("----------------------")
        txt = xpdf_python.to_text(myfile)[0]
        txt_arr = txt.strip().split("\n")
        txt_arr = [x.strip() for x in txt_arr]
        txt_arr = [x.replace("\r"," ") for x in txt_arr]
        txt_arr = [myfile_short+" : " + x + "\n" for x in txt_arr]
        txt = "".join(txt_arr)
        ss += txt
        print(txt)

    print("----------------------")
    print("writing to file ", fname_out)
    fh = open(fname_out, "w")
    fh.write(ss)
    fh.close()
    print("----------------------")
