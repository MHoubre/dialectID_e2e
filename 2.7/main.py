import sys
from utils import run_on_dataset
from display_results import get_references, general_display
import glob2

if __name__=="__main__":

    path= sys.argv[1] # we get the path of the dataset
    print("Launching the algorithm on the dataset situated in " +path)
    FILENAME = glob2.glob(path+'*.wav', recursive=True) # returns a list with all the wav files in your data folder. Using glob2 https://github.com/miracle2k/python-glob2
    FEAT_TYPE = 'logmel'
    

    languages= {'EGY':0,'GLF':1, 'LAV':2, 'MSA':3, 'NOR':4} 

    references = get_references(FILENAME, languages)
    predictions = run_on_dataset(FILENAME,FEAT_TYPE)

    general_display(references, predictions, labels= languages ,
                    class_report = True, report_to_csv=True, conf_matrix = True)

    #print(results)