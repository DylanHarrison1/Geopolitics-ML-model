from model import Model
import pandas as pd
import os
import numpy as np
from preprocessing import RemoveColumns
import matplotlib.pyplot as plt

class Instance():
    def __init__(self, feedback):
        """
        Input:
        feedback data, do we want feedback, Binary - 
        Model Details, Int - 
        dbList, list of data used - 
        
        """
        self._feedback = feedback
        self._instance = Model(5)
        self._lossData = []

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
        

        for i in range(epochs):
            #Open relevant csv files here
            Demog = pd.read_csv(os.getcwd() + '\data\\processed\Demographics.csv')
            LDI = pd.read_csv(os.getcwd() + '\data\\processed\HDI (1870-2020).csv')
            HDI = pd.read_csv(os.getcwd() + '\data\\processed\Liberal Democracy Index.csv')
            
            #loses unecesary years
            Demog.drop(Demog.columns[range(73, 80)], axis=1, inplace=True)
            LDI.drop(LDI.columns[range(147, 152)], axis=1, inplace=True)
            LDI.drop(LDI.columns[range(1,81)], axis=1, inplace=True)
            HDI.drop(HDI.columns[range(1,81)], axis=1, inplace=True)

            self.__CountryLoop(Demog, LDI, HDI, False)
    
        #Saves model for further testing
        parameters = []
        for name, param in self._instance.named_parameters():
            parameters.append({'Name': name, 'Value': param.data.numpy()})

        df = pd.DataFrame(parameters)
        df.to_csv('model_parameters.csv', index=False)

        self.__TestModel(Demog, LDI, HDI)
        self.__PrintGraph()
        
                    

    def __CountryLoop(self, Demog, LDI, HDI, Testing):
        
            score = []

            #loops through all coutries in Demography
            for j in range(0, 5544, 99):
                x1 = Demog.iloc[[j + 8, j + 93, j + 94, j + 95], range(7, 73)]
                x2 = self.__DemogToHDI_LDI(LDI, Demog.iloc[(j,0)])
                x = pd.concat([x1, x2], ignore_index=True)
                y = self.__DemogToHDI_LDI(HDI, Demog.iloc[(j,0)])

                if not isinstance(y, pd.DataFrame):
                    continue
                elif y.isna().any().any():
                #Checks all cells have some value
                    continue
                y = y.values

                #Loops through all years for country
                lossMean = 0

                if Testing:
                        k = x.shape[1] - 1
                        yPred = self._instance.calc(x.iloc[:,k]).detach().numpy()
                        yAct = y[0, range(k, k+5)]
                        
                        #fixes relative closeness between 0 and 1
                        relCloseness = [abs(yAct[i]/ yPred[i]) for i in range(5)]
                        for i in range(5):
                            if (relCloseness[i] > 1):
                                relCloseness[i] = 1 / relCloseness[i]
                        score.append(relCloseness)

                else:
                    
                    #Main training
                    for k in range(x.shape[1] - 5):
                        yPred = self._instance.calc(x.iloc[:,k])
                        loss, gradient = self._instance.train(yPred, y[0, range(k, k+5)]) 
                        lossMean += loss

                    lossMean /= x.shape[1]
                    self._lossData.append(lossMean)
                    if self._feedback:
                        
                        self.__PrintProgress(j, lossMean, gradient)

            if Testing:
                total = [0,0,0,0,0]
                for i in score:
                    for j in range(len(total)):
                        total[j] += i[j]
                total = [i / len(score) for i in total] 
                print(total) 
                print(score)


    def __PrintGraph(self):
        data = [i.detach().numpy() for i in self._lossData]
        x = np.arange(len(data))
        m, b = np.polyfit(x, data, 1)

        
        plt.figure(figsize=(8, 5))
        plt.scatter(x, data, color='green', label='Data Points')
        plt.plot(x, m*x + b, color='red', label='Line of Best Fit')
        #Stop outliers from distorting the graph
        plt.ylim(min(data), max(data) * 0.5)
        
        plt.xlabel('Time (51 units=1 epoch)')
        plt.ylabel('Average Loss')
        plt.title('Loss over Time')

        plt.legend()
        plt.grid(True)
        plt.show()

    def __TestModel(self, Demog, LDI, HDI):
        self.__CountryLoop(Demog, LDI, HDI, True)

    def __PrintProgress(self, j, lossMean, gradient):
        print(j/99)
        print("~~~~~loss~~~~~")
        print(lossMean)
        #print("~~~~~latest gradient~~~~~")
        #print(gradient[0])
    
