import pandas as pd
import os
#Goals:
# - 
# - Create A HDI file per country per year
# - Simplify 
# - 
# - 
# - 

def FetchData(row, collumn, directory):
    curDir = os.getcwd()
    df = pd.read_csv(curDir + directory)
    print(df.iloc[(0,3)])


def CreateHDIFile():
    #spaghetti logic, but should add in blank columns for missing dates.
    df = pd.read_csv(os.getcwd() + '..\data\raw\AHDI (1870-2020) (excl income).csv')
    i = 1
    while (True):
        dif = df.iloc[(0,i + 1)] - df.iloc[(0,i)]
        for j in range(1, dif):
            a = [''] * 168
            a[0] = i + j
            df.insert(i, '', a, allow_duplicates=True)
            i += 1
        i += 1
    
def Simplify_xxxx_File():
    pass


FetchData(1, None, '\data\\raw\AHDI (1870-2020) (excl income).csv')