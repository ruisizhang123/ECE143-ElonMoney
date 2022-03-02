# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 23:22:03 2022

@author: Yen
"""
from getDict import generateSet


from monkeylearn import MonkeyLearn
 
ml = MonkeyLearn('24ab34508e23db344ab42a853007d54903f4968f')
def get_analysis(text):
    data = [text]
    model_id = 'cl_pi3C7JiL'
    result = ml.classifiers.classify(model_id, data)

    return result.body[0]


def createData(startID, endID, savingName):
    '''
    using inbuilt dictionary to extract desired data from startID row to endID row.

    '''
    assert startID > 1
    
    # inbuilt list
    l = ['AutoDriving', 'BitCoin', 'CleanEnergy', 'DogeCoin', 'Ecar', 'Ethereum', 'Rocket']
    key_list = generateSet(l)
    
    import csv
    file = open('elcon.csv', encoding="cp437")
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    currentID = 2
    
    rows = []
    for row in csvreader:
        if currentID < startID:
            currentID += 1
            continue
        elif currentID > endID:
            break
        
        context = row[10].lower()
        
        for pair in key_list:
            if any(x in context for x in pair[1]):
                #print('{} is {}'.format(currentID, pair[0]))
                new_time_format = row[2]
                time_list = new_time_format.split('-')
                date = time_list[2].split(' ')
                
                new_time_format = time_list[0] + '/' + time_list[1] + '/' + date[0]
                
                # do sentiment analysis
                result = get_analysis(context)
                rows.append([currentID, pair[0],new_time_format, context, result['classifications'][0]['tag_name'], result['classifications'][0]['confidence']])
        currentID += 1
    file.close()
    # save list into file
    fields = ['ID', 'Class', 'Time', 'Context', 'Sentiment', 'Confidence']
    filename = savingName + '.csv'
    # writing to csv file 
    with open(filename, 'w', newline='', encoding="utf-8") as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(rows)
    
    return rows

        