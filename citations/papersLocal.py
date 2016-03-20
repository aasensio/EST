import ads
import csv
import numpy as np
from collections import Counter
from ipdb import set_trace as stop
import pickle

def initials(str):
    return '.'.join(name[0].upper() for name in str.split())+'.'


csvfile = open('names.csv', 'r')
reader = csv.reader(csvfile, delimiter=',')

try:
    with open('outputKeywords.csv') as foo:
        nlines = len(foo.readlines())
except:
    nlines = 0

if (nlines == 0):
    outFile = open('outputKeywords.csv', 'w')
else:
    outFile = open('outputKeywords.csv', 'a')
 
writer = csv.writer(outFile, delimiter=',')

database = pickle.load(open("database.dat", "rb"))

loop = 1
for row in reader:
    print(row)
    if (loop > nlines):
        name = "{0}, {1}".format(row[1], initials(row[0]))
        print("{0}".format(name))

        info = database[name]

        nPapers = len(info)
        print("  N. papers: {0}".format(nPapers))
        if (nPapers > 0):
            years = np.asarray([int(info[i].year) for i in range(nPapers) if (info[i].year != None)])        
            print("  Years: {0} - {1}".format(np.min(years), np.max(years)))
        else:
            years = np.asarray([0,0])

    # Get the keywords and get the three more abundant
        print("  Getting keywords")
        temp = np.asarray([info[i].keyword for i in range(nPapers) if info[i].keyword != None])
        keywords = [item for sublist in temp for item in sublist]
        mcommon= [ite for ite, it in Counter(keywords).most_common(3)]    
        if (not mcommon):
            mcommon = [None, None, None]
        else:
            print("  Keywords: {0}, {1}, {2}".format(mcommon[0], mcommon[1], mcommon[2]))

        writer.writerow([name, nPapers, np.min(years), np.max(years), mcommon[0], mcommon[1], mcommon[2]])
        outFile.flush()
    loop += 1
    

csvfile.close()
outFile.close()