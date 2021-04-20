
"""
# pdf2txt.py
# script recursively extractis text from all PDF files
# and saves to a text file 
# It looks either in current directory
# or in directories provided on command line (space-separaed)
""" 

# pip install PyPDF2
import os, sys, glob
import PyPDF2 

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

        pdfFileObj = open(myfile, 'rb') 
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
        N_pages = pdfReader.numPages
        for p_num in range(N_pages):
            pageObj = pdfReader.getPage(p_num) 
            txt = pageObj.extractText()
            txt_arr = txt.strip().split("\n")
            txt_arr = [x.strip() for x in txt_arr]
            txt_arr = [x.replace("\r"," ") for x in txt_arr]
            txt_arr = [myfile_short+" : " + x + "\n" for x in txt_arr]
            txt = "".join(txt_arr)
            ss += txt
            print(txt)
        pdfFileObj.close() 

    print("----------------------")
    print("writing to file ", fname_out)
    fh = open(fname_out, "w")
    fh.write(ss)
    fh.close()
    print("----------------------")
