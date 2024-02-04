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
    #df = pd.read_csv('..\data\raw\AHDI (1870-2020) (excl income).csv')
    #while (True):

    #    df.insert()
    pass
def Simplify_xxxx_File():
    pass


FetchData(1, None, '\data\\raw\AHDI (1870-2020) (excl income).csv')