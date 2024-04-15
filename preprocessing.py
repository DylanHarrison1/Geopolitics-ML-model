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


def ReadDF(path: str) -> pd.DataFrame:
    """
    Locates dataframe in local files and returns.
    """
    
    path = os.getcwd() + path
    df = pd.read_csv(path)
    return df

def AddEmptyColumns(file: str, interpolate: bool) -> None:
    '''
    Designed for DB's ordered by year. 
    If some years are missing, it will add them in with 
    blank values or linearly interpolated values (if 
    "interpolate" is True or False")
    '''

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

def InterpolateColumn(df: pd.DataFrame, year1: int, year2: int, newYear: int) -> pd.DataFrame:
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
    '''
    Orders columns (excluding first) by date
    '''
    
    column0 = df.columns[0]

    mainColumns = sorted([col for col in df.columns if col.isdigit()])
    new_columns = [column0] + mainColumns
    df = df[new_columns]
    return df

def Remove_LDI():
    '''
    #Removes the Liberal democracy index from the AHDI
    #AHDI is the geometric mean of 4 values.
    #Therefore, we ^4, then divide by LDI, then cube root.
    '''
   
    AHDI = pd.read_csv(os.getcwd() + '\data\\raw\pure AHDI (1870-2020).csv', encoding='latin-1')
    LDI = pd.read_csv(os.getcwd() + '\data\\raw\Liberal Democracy Index.csv', encoding='latin-1')

    result = (AHDI.iloc[:, 1:] ** 4 / LDI.iloc[:, 1:]).apply(np.cbrt)
    result.insert(0, AHDI.columns[0], AHDI.iloc[:, 0])
    
    result.to_csv(os.getcwd() + '\data\\raw\HDI (1870-2020).csv', encoding='latin-1', index=False)

def RemoveColumns(path, columns):
    '''
    Removes specified list of columns (index from 0)
    '''
    path = os.getcwd() + path
    df = pd.read_csv(path)
    df.drop(df.columns[columns], axis=1, inplace=True)
    df.to_csv(path, index=False)

def rearrangeOECD(inputPath, outputPath):
    '''
    Makes OECD demographics more compact by turning 400000 rows into like 5000*70
    '''

    inputPath = os.getcwd() + inputPath
    outputPath = os.getcwd() + outputPath
    df = pd.read_csv(inputPath)

    years = list(range(1950, 2023))
    for year in years:
        df[year] = None

    # Goes through df in blocks of 72
    for i in range(0, len(df), 73):
        block = df.iloc[i:i+73]  
        blockValues = block['OBS_VALUE'].values
        df.loc[i, years] = blockValues  # Put the values into the block's top row

    # Keep only the first row of each block of 72 rows
        
    df = df.iloc[::73]
    df.to_csv(outputPath, index=False)

def OrderCSVRows(path, columnNumbers):
    '''
    Orders CSV rows alphabetically by certain columns.
    List is inputted of column precedence.
    '''
    path = os.getcwd() + path
    df = pd.read_csv(path)
    sorted_df = df.sort_values(by=[df.columns[i] for i in columnNumbers])
    
    sorted_df.to_csv(path, index=False)


def TemplateVDem(path: str) -> None:
    """
    Works, creates a table as template for filling for V-Dem.
    """
    olddf = ReadDF(path)

    countries = olddf['country_name'].unique()
    years = olddf['year'].unique()
    measures = olddf.columns[2:]

    
    finc = [country for country in countries for i in range(len(measures))]

    finm = [measure for i in range(len(countries)) for measure in measures ]
    
    years = [str(i) for i in years]

    years = ['country_name','indices'] + years

    
    df = pd.DataFrame(columns=years)
    df['country_name'] = finc
    df['indices'] = finm
    df.to_csv(os.getcwd() + "\\test.csv", index=False)

def TemplateVParty(path: str) -> None:
    olddf = ReadDF(path)

    countries = olddf['country_name'].unique()
    years = range(1900, 2024)
    x = np.array(olddf[['country_name','v2paenname']], dtype=str)
    #x = list(olddf[['v2paenname','country_name']])
    parties = np.unique(x, axis=0)
    #print(parties[:,0])
    measures = olddf.columns[3:]
    
    
    finp = [party for party in parties for i in range(len(measures))]
    finm = [measure for i in range(len(parties)) for measure in measures ]
    years = [str(i) for i in years]

    years = ['country_name','v2paenname','indices'] + years

    
    df = pd.DataFrame(columns=years)
    df['country_name'] = [i[0] for i in finp]
    df['v2paenname'] = [i[1] for i in finp]
    df['indices'] = finm
    df.to_csv(os.getcwd() + "\\test.csv", index=False)


def FillVDem(oldpath: str, newpath: str) -> None:
    olddf = ReadDF(oldpath)
    df = ReadDF(newpath)
    #27734 vdem
    for j in range(1, 27734):
        #Finds equivilant cell in new df, using country, index, and year
        #targets = df.loc[df['country_name'] == olddf.iloc[(j,0)] , str(olddf.iloc[(j,1)])]
        
        row = df.index[df['country_name'] == olddf.iloc[(j,0)]]
        column = str(olddf.iloc[(j,1)])

        #print(targets)
        for k in range(len(row)):
            df.loc[row[k], column] = olddf.iloc[(j, k + 2)]
            
            #and df['indices'] == olddf.columns[i]
    df.to_csv(os.getcwd() + "\\test2.csv", index=False)

def FillVParty(oldpath: str, newpath: str) -> None:

    olddf = ReadDF(oldpath)
    df = ReadDF(newpath)
    #11899 vparty
    for j in range(1, 11898):
        #Finds equivilant cell in new df, using country, index, and year
        #targets = df.loc[df['country_name'] == olddf.iloc[(j,0)] , str(olddf.iloc[(j,1)])]
        
        row1 = df.index[df['country_name'] == olddf.iloc[(j,1)]]
        row2 = df.index[df['v2paenname'] == olddf.iloc[(j,0)]]
        row = [i for i in row1 if i in row2]

        
        column = str(olddf.iloc[(j,2)])

        #print(targets)
        for k in range(len(row)):
            df.loc[row[k], column] = olddf.iloc[(j, k + 3)]
            
            #and df['indices'] == olddf.columns[i]
    df.to_csv(os.getcwd() + "\\test2.csv", index=False)




'''
Code used to run functions
'''
#FetchData(1, None, '\data\\raw\AHDI (1870-2020) (excl income).csv')
#Remove_LDI()
    
#AddEmptyColumns('\data\\raw\pure AHDI (1870-2020) copy.csv', True)

#Action, TIME_HORIZ, Time Horizon, 
#RemoveColumns('\data\\raw\OECD population by sex, age range.csv', [])

#rearrangeOECD('\data\\raw\OECD population by sex, age range.csv', '\data\\raw\Demographics.csv')
#RemoveColumns('\data\\raw\Demographics.csv', [5, 6])

#OrderCSVRows('\data\\raw\Demographics.csv', [0, 1, 4, 3, 2])
#OrderCSVRows('\data\\raw\HDI (1870-2020).csv', [0])

#AddEmptyColumns('\data\\raw\HDI (1870-2020).csv', True)
#OrderCSVRows('\data\\processed\Demographics.csv', [0, 1, 2, 4, 3])
#AddEmptyColumns('\data\\raw\Liberal Democracy Index.csv', True)


#FillV('\data\\raw\V-Dem\V-Dem Core High Level Indices.csv', '\\test.csv')

#TemplateVParty('\data\\raw\V-Dem-CPD-Party-V2.csv')

#FillVParty('\data\\raw\V-Dem-CPD-Party-V2.csv','\\test.csv')