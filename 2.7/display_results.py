import seaborn as sn
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def get_references(filelist,labels):
    
    references= []

    for filename in filelist:
        for key in labels.keys():
            if key in filename:
                references.append(labels[key])
                continue
    return references


def plot_classification_report(y_true, y_pred, labels):

    report = classification_report(y_true, y_pred, target_names= labels.keys(), output_dict=True)

    # For each language, we have a dict with the different metrics
    # We get the keys of the first subdict
    metriclabels= report[next(iter(report))].keys()
    

    
    # A matrix that will get all the values for each key of the report
    matrix = []
    
    for i,key in enumerate(labels.keys()): # for each category
        metrics=[0]*len(metriclabels)
        for j,metric in enumerate(metriclabels): # for each metric
            metrics[j] = report[key][metric]    # we get the value 
        matrix.append(metrics) # we add the metrics of the category in the list

    # We plot
    plt.figure(figsize=(11,7))
    hm= sn.heatmap(matrix, annot= True, xticklabels=metriclabels, yticklabels=labels.keys(), cmap="YlGnBu")

    plt.show()
    figure= hm.get_figure()
    figure.savefig("images/"+str(date.today())+"_report-heatmap.png")
    
    return report

def plot_confusion_matrix(y_true, y_pred, labels):
    # We revert the dictionnary so that the indexes are the keys and the langue names are the values
    inv_map = {value: key for key, value in labels.iteritems()}

    # We want outputs like with the name of the language, not the index in the dictionnary
    inv_y_true = [inv_map[value] for value in y_true]
    
    inv_y_pred= [inv_map[value] for value in y_pred]
    
    fig, ax= plt.subplots()
    matrix = confusion_matrix(inv_y_true, inv_y_pred, labels=labels.keys())
    hm= sn.heatmap(matrix, annot= True, xticklabels=labels.keys(), yticklabels=labels.keys())

    figure= hm.get_figure()
    figure.savefig("images/"+str(date.today())+"_confusion-matrix.png")

    return



def general_display(y_true, y_pred, labels ,class_report = True, report_to_csv=False, conf_matrix = True):
    
    if class_report:

        report = plot_classification_report(y_true, y_pred, labels)

        if report_to_csv:

            dataframe = pd.DataFrame.from_dict(report, orient="index")

            dataframe.to_csv(str(date.today())+".csv")
            print(dataframe)


    if conf_matrix:

        plot_confusion_matrix(y_true, y_pred, labels)


