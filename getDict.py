# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 23:48:49 2022

@author: Yen
"""

def generateSet(l):
    '''
    return list of [subject, keyword_set]

    example: generateSet(['Ecar', 'Rocket'])
    
    path requirement:
        -getDict.py
        -keywords
         --Ecar.txt
         --Rocket.txt

    '''
    parent_path = 'keywords/'
    return_list = []
    for subject in l:
        path = parent_path + subject + '.txt'
        file = open(path)
        s = set()
        for text in file:
            s.add(text.rstrip('\n').lower())
        return_list.append([subject, s])
    return return_list
        