import PyPDF2
from PyPDF2 import PdfFileReader
import pycpdf
import urllib3
import os
import re

ftfile = 'text.txt'

http = urllib3.PoolManager()
r = http.request('Get',"http://www.gutenberg.org/cache/epub/10/pg10.txt",preload_content=False)
with open(ftfile,'wb') as out:
    while True:
        data = r.read()
        if not data:
            break
        out.write(data)
r.release_conn()

fulltext = open(ftfile,'r')
mystr = fulltext.read()
# mystr = mystr.lower()

groups = re.findall('(.*\n+1\:1\s(.+\n{1,2})+)',mystr)

composite = open('fulltext.txt','w')
for g in groups:
    mysplit = re.split('\n',g[0],1)
    finalout = open(mysplit[0]+".txt",'w')
    nonum = re.sub("(\d+\:\d+)","AAAAAA",mysplit[1])
    #noline = nonum.replace('\n','SSSSSS ')
    # noline = re.sub('\n+',"\ ",nonum)
    finalout.write(nonum)
    composite.write(noline)
    finalout.close()
composite.close()
print('sdsd')
    
