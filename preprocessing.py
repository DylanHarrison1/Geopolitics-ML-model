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

'''
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
        '''

    
def AddEmptyColumns(file, interpolate):
    filename = os.getcwd() + file
    df = pd.read_csv(filename)

    # Get list of column names
    years = df.columns[1:]
    years = [int(i) for i in years]

    newColumns = []

    #Find difference between 2 years, add that many columns
    for i in range(len(years) - 1):
        dif = years[i+1] - years[i] - 1
        if dif > 0:
            for j in range(dif):
                newYear = years[i] + j + 1

                if interpolate:
                    df[str(newYear)] = df[[str(years[i]), str(years[i+1])]].interpolate(axis=1)[str(newYear)]
                else:
                    newColumn = pd.Series("", name=str(newYear), index=df.index)
                    newColumns.append(newColumn)

    df = pd.concat([df] + newColumns, axis=1)
    df = OrderByDate(df)
    df.to_csv(filename, index=False)

def OrderByDate(df):
    #Orders columns (excluding first) by date
    column0 = df.columns[0]

    mainColumns = sorted([col for col in df.columns if col.isdigit()])
    new_columns = [column0] + mainColumns
    df = df[new_columns]
    return df

def Remove_LDI():
    #Removes the Liberal democracy index from the AHDI
    #AHDI is the geometric mean of 4 values.
    #Therefore, we ^4, then divide by LDI, then cube root.
    AHDI = pd.read_csv(os.getcwd() + '\data\\raw\pure AHDI (1870-2020).csv', encoding='latin-1')
    LDI = pd.read_csv(os.getcwd() + '\data\\raw\Liberal Democracy Index.csv', encoding='latin-1')

    result = (AHDI.iloc[:, 1:] ** 4 / LDI.iloc[:, 1:]).apply(np.cbrt)
    result.insert(0, AHDI.columns[0], AHDI.iloc[:, 0])
    
    result.to_csv(os.getcwd() + '\data\\raw\HDI (1870-2020).csv', encoding='latin-1', index=False)



#FetchData(1, None, '\data\\raw\AHDI (1870-2020) (excl income).csv')
#Remove_LDI()
    
AddEmptyColumns('\data\\raw\pure AHDI (1870-2020) copy.csv', True)