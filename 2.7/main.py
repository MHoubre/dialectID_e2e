import sys, os
from utils import run_on_dataset
from sklearn.metrics import classification_report
import glob2

def get_references(filelist,labels):
    
    references= []

    for filename in filelist:
        for key in labels.keys():
            if key in filename:
                references.append(labels[key])
                continue
    return references


if __name__=="__main__":

    FILENAME = glob2.glob('./data/**/*.wav', recursive=True) # returns a list with all the wav files in your data folder. Using glob2 https://github.com/miracle2k/python-glob2
    languages= {'EGY':0,'GLF':1, 'LAV':2, 'MSA':3, 'NOR':4} 
    FEAT_TYPE = 'logmel'
    print(FILENAME)
    
    references = get_references(FILENAME,languages)
    results = run_on_dataset(FILENAME,FEAT_TYPE)

    print(classification_report(references, results, target_names= languages.keys())

    #print(results)
