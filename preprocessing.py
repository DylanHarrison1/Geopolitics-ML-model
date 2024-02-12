import pandas as pd
import os
import numpy as np
#Goals:
# - 
# - Create A HDI file per country per year
# - Similarly format other files 
# - 
# - 
# - Create dataset per country with inputs and outputs over time.

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
    
def Remove_LDI():
    #Removes the Liberal democracy index from the AHDI
    #AHDI is the geometric mean of 4 values.
    #Therefore, we ^4, then divide by LDI, then cube root.
    AHDI = pd.read_csv(os.getcwd() + '\data\raw\AHDI (1870-2020) (excl income).csv')
    LDI = pd.read_csv(os.getcwd() + '\data\raw\Liberal Democracy Index.csv')
    HDI = pd.read_csv(os.getcwd() + '\data\rawLiberal Democracy Index.csv')

    result_df = (AHDI ** 4 / LDI).apply(np.cbrt)
    
    result_df.to_csv(HDI, index=False)

    #method to apply to a whole row/ whole table?
    #Maybe extract into array, then math


#FetchData(1, None, '\data\\raw\AHDI (1870-2020) (excl income).csv')
Remove_LDI()