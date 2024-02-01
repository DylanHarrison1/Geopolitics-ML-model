import csv

#Goals:
# - Create A HDI file per country per year
# - Simplify 
# - 
# - 
# - 

def CreateHDIFile():
    with open('data\raw\AHDI (1870-2020) (excl income).csv', newline='') as csvfile:
        dataReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in dataReader:
            print(', '.join(row))


def Simplify_xxxx_File():
    pass
