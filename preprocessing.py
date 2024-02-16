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
                    newColumn = InterpolateColumn(df, years[i], years[i+1], newYear)
                else:
                    newColumn = pd.Series("", name=str(newYear), index=df.index)
                
                newColumns.append(newColumn)

    df = pd.concat([df] + newColumns, axis=1)
    df = OrderByDate(df)
    df.to_csv(filename, index=False)

def InterpolateColumn(df, year1, year2, newYear):
    '''
    Interpolates across years to estimate missing values
    '''

    newValues = []
    for i in range(df.shape[0]):
        value1 = df.loc[i, str(year1)]
        value2 = df.loc[i, str(year2)]
        
        # If either value is NaN, skip interpolation
        if pd.isna(value1) or pd.isna(value2):
            newValues.append(None)
        else:
            # Perform linear interpolation
            ratio = (newYear - year1) / (year2 - year1)
            newValue = value1 + (value2 - value1) * ratio
            newValues.append(newValue)
    
    # Add new column to the dataframe
    newColumn = pd.Series(newValues, name=str(newYear))
    return newColumn

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

def RemoveColumns(path, columns):
    path = os.getcwd() + path
    df = pd.read_csv(path)
    df.drop(df.columns[columns], axis=1, inplace=True)
    df.to_csv(path, index=False)


'''
Code used to run functions
'''
#FetchData(1, None, '\data\\raw\AHDI (1870-2020) (excl income).csv')
#Remove_LDI()
    
#AddEmptyColumns('\data\\raw\pure AHDI (1870-2020) copy.csv', True)

#Action, TIME_HORIZ, Time Horizon, 
RemoveColumns('\data\\raw\OECD population by sex, age range.csv', [0,])