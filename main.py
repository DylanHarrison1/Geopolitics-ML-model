from model import Model
import pandas as pd
import os
import numpy as np
from preprocessing import RemoveColumns


class Instance():
    def __init__(self, feedback, dbListIn, dbListOut):
        """
        Input:
        feedback data, do we want feedback, Binary - 
        Model Details, Int - 
        dbList, list of data used - 
        
        """
        self._feedback = feedback
        self._instance = Model(20)

    def __DemogToHDI_LDI(self, df, key):
        '''
        Fetches corresponding data to Demog from the HDI and LDI
        '''
        if key in df.iloc[:, 0].values:
            #returns the row excluding its name
            return df.loc[df.iloc[:, 0] == key].iloc[:, 1:]
        else:
            return None

    def Run(self, epochs):
        """
        Input:
        epochs, Int - 

        """
        

        for i in epochs:
            #Open relevant csv files here
            Demog = pd.read_csv(os.getcwd() + '\data\\processed\Demographics.csv')
            LDI = pd.read_csv(os.getcwd() + '\data\\processed\HDI (1870-2020).csv')
            HDI = pd.read_csv(os.getcwd() + '\data\\processed\Liberal Democracy Index.csv')
            
            #loses unecesary years
            Demog.drop(Demog.columns[range(73, 80)], axis=1, inplace=True)
            LDI.drop(LDI.columns[range(147, 152)], axis=1, inplace=True)
            LDI.drop(LDI.columns[range(1,81)], axis=1, inplace=True)
            HDI.drop(HDI.columns[range(1,81)], axis=1, inplace=True)


            for j in range(0, 5544, 99):
                x = self.__DemogToHDI_LDI(LDI, Demog.iloc[(i,0)])

                y = self.__DemogToHDI_LDI(HDI, Demog.iloc[(i,0)])
                if y == None:
                    continue

                yPred = self._instance.calc("??????x")
                self._instance.train(yPred, "y") 

                if self._feedback:
                    self.__PrintProgress()

                #When csv files are depleted, close them.
                

        #Final data print goes here
                    

    def __PrintProgress():
        pass
    
