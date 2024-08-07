from model import Model, TempConvNet
import pandas as pd
import os
import numpy as np
from preprocessing import RemoveColumns
from preprocessing import ReadDF
import matplotlib.pyplot as plt
import torch
#import keras
from tcn import TCN
    
    


class Instance():

    def __init__(self, modelType: str, modelStructure: list, datasets: list, indices: list, combMethod: str, trainLength: int, yrToPredict: int, feedback: bool = False, graph: bool = False) -> object:
        """
        modelType, modelStructure- inputs for model[__]
        datasets- list of datasets used to train[__]
        indices- which indices are we using from each dataset[__]
        combMethod- how are we combining datasets? extrapolate, min, train[__]
        trainLength- How many years worth of data are we using as input (only for TCN)[__]
        yrToPredict- How many years are being predicted?
        feedback- do we want feedback? [__]
        graph- Dd we want to graph the result?
        """
        torch.random.manual_seed(1)
        self._feedback = feedback
        self._graph = graph
        self._modelType = modelType
        self._lossData = []
        self._modelType = modelType
        self._modelStructure = modelStructure
        self._indices = indices
        self._trainLength = trainLength
        self._yrToPredict = yrToPredict

        #Gets the subset of meta with just these datasets
        meta = ReadDF("\data\\processed\meta.csv", None)

        meta = meta[meta.loc[:, "DBName"].isin(datasets)]

        #Puts all data in the data list
        self._data = []

        for i in range(meta.shape[0]):
            self._data.append(ReadDF(meta.iloc[i,1], None))

        #Fits the datasets to each other
        if (combMethod == "slice"):

            #Slicing years
            top = int(meta["YrEnd"].min())
            bottom = int(meta["YrStart"].max())
            for df in self._data:

                colList = [int(col) for col in df.columns if col.isdigit()]
                toKeep = [col for col in colList if bottom <= col <= top]

                toDrop = set(colList) - set(toKeep)
                toDrop = [str(col) for col in toDrop]

                df.drop(columns= toDrop, inplace= True)

            #Slicing Countries
            commonCountries = set(self._data[0].iloc[:, 0])
            for df in self._data:  
                commonCountries = commonCountries.intersection(df.iloc[:, 0])
            for df in self._data:  
                df.drop(df[~df.iloc[:, 0].isin(commonCountries)].index, inplace=True)
        
        #Loses index names, only a hindrance now
        for i in range(len(self._data)):
            columns = range(int(meta.iloc[i, 2]))
            self._data[i].drop(self._data[i].columns[columns], axis=1, inplace=True)

        self._meta = meta

        #Creates Model
        if modelType == "basic":
            self._instance = Model(modelStructure)
        elif modelType == "TCN":
            #self._trainLength = self._data[0].shape[1] - (yrToPredict * 2)
            self._trainLength = trainLength
            self._instance = TempConvNet(self._trainLength, 31, modelStructure)
          
        elif modelType == "TCN+":
            pass
            self._trainLength = trainLength
            for data in self._data:
                pass

    def Run(self, epochs):
        """
        Input:
        epochs, Int - 

        """

        for i in range(epochs):
            #print(i)
            self.__CountryLoop()
    
        #Saves model for further testing
        parameters = []
        for name, param in self._instance.named_parameters():
            parameters.append({'Name': name, 'Value': param.data.numpy()})

        df = pd.DataFrame(parameters)
        df.to_csv('model_parameters.csv', index=False)

        if self._graph:
            self.__PrintGraph()           

    def __CountryLoop(self):
        predYears = self._modelStructure[-1] #output size
        lossMean = 0


        for i in range(self._data[0].shape[0]): #Loop through all countries
            x = self.__GetX(i)
            

            if self._modelType == "basic":
                stop = self._data[0].shape[1] - (predYears * 2)
                for j in range(stop): #Loops to year end minus test set

                    thisx = [this[j] for this in x]
                    yAct = self._data[0].iloc[i, range(j, j + predYears)]
                    
                    yPred = self._instance.forward(thisx)
                    yPred = self.__AddGaussianNoise(yPred)

                loss, gradient = self._instance.train(yPred, yAct) 
                lossMean += loss

            elif self._modelType == "TCN":
    
                for j in range(self._data[0].shape[1] - self._trainLength - (self._yrToPredict)):
                    startIndex = j
                    stopIndex = j + self._trainLength    
                    startPred = startIndex + self._yrToPredict
                    stopPred = stopIndex + self._yrToPredict

                    trainto = self._data[0].shape[1] - predYears


                    thisx = [this[startIndex:stopIndex] for this in x]
                    yAct = self._data[0].iloc[i, startPred:stopPred]
                    #thisx = np.array(thisx).T


                    yPred = self._instance.forward(thisx)
                    #print("ypred " + str(yPred))
                    yPred = self.__AddGaussianNoise(yPred)
                    
                    #print("yAct " + str(yAct))
                    
                    loss, gradient = self._instance.train(yPred, yAct) 
                    lossMean += loss
                    #print("loss" + str(loss))
                    #print("gradient" + str(gradient))

            if self._graph:
                lossMean /= len(x)
                self._lossData.append(lossMean)  
            if self._feedback:
                self.__PrintProgress(i, self._data[0].shape[0], lossMean, gradient)

    def __PrintGraph(self):
        data = [i.detach().numpy() for i in self._lossData]
        x = np.arange(len(data))
        m, b = np.polyfit(x, data, 1)

        
        plt.figure(figsize=(8, 5))
        plt.scatter(x, data, color='green', label='Data Points')
        #plt.plot(x, m*x + b, color='red', label='Line of Best Fit')
        #Stop outliers from distorting the graph
        plt.ylim(min(data), max(data) * 0.5)
        
        plt.xlabel('Training Progress (1 unit=1 country) (51 units=1 epoch)')
        plt.ylabel('Average Loss')
        plt.title('Loss over Time')

        plt.legend()
        plt.grid(True)
        plt.show()

    def TestModel(self):
        
        if self._modelType == "basic":
            predYears = self._modelStructure[-1]  
            score = [] 
            for i in range(self._data[0].shape[0]): #Loop through all countries
                x = self.__GetX(i)

                
                start = self._data[0].shape[1] - (predYears * 2)
                stop = self._data[0].shape[1] - predYears
                    
                for j in range(start, stop): #last years
                    
                    thisx = [this[j] for this in x]
                    yPred = self._instance.forward(thisx)
                    yAct = self._data[0].iloc[i, range(j, j + predYears)]


                    relCloseness = [abs(yAct[k]/ yPred[k]) for k in range(predYears)]
                    for k in range(predYears):
                        if (relCloseness[k] > 1):
                            relCloseness[k] = 1 / relCloseness[k]
                    score.append(relCloseness)
            
            total = [0] * predYears
            for i in score:
                for j in range(predYears):
                    total[j] += i[j]
            total = [i / len(score) for i in total] 
            return total   
          
        elif self._modelType == "TCN":
            startIndex = self._data[0].shape[1] - self._trainLength - (self._yrToPredict)
            stopIndex = startIndex + self._trainLength

            predYears = (self._data[0].shape[0] - 5, self._data[0].shape[0])
            score = []  
            for i in range(self._data[0].shape[0]): #Loop through all countries
                x = self.__GetX(i)

                trainfrom = self._yrToPredict * 2
                
                thisx = [this[startIndex:stopIndex] for this in x]
                yPred = self._instance.forward(thisx)
                yAct = torch.tensor(self._data[0].iloc[i, (startIndex +self._yrToPredict):(stopIndex + self._yrToPredict)])
                
                loss = torch.nn.functional.mse_loss(yPred, yAct)
                #print(loss)
                #print("ypred " + str(yPred))
                #print("yact " + str(yAct))
                #print(yPred)
                #print(yPred.shape, yAct.shape)
                
                relCloseness = [abs(yAct[k]/ yPred[k]) for k in range(self._trainLength - trainfrom, yPred.shape[0])]
                for k in range(len(relCloseness)):
                    if (relCloseness[k] > 1):
                        relCloseness[k] = 1 / relCloseness[k]
                score.append(relCloseness)

            #print(score)

            total = [0] * self._yrToPredict
            for i in score:
                for j in range(self._yrToPredict):
                    total[j] += i[j]
            total = [i / len(score) for i in total] 
            return total

    def __PrintProgress(self, i, countries, lossMean, gradient):
        print("~~~~~~~~~~~~~~~~~~~~~~~~")
        print(str(i) + "/" + str(countries) + ". Mean Loss = " + str(lossMean))

        gradmean = torch.mean(gradient[0])
        print("Latest Gradient Mean = " + str(gradmean))
    
    def __AddGaussianNoise(self, data, mean=0, std=0.001):
        noise = torch.randn(data.size()) * std + mean
        return data + noise
    
    def __GetX(self, i: int) -> list:
        """
        Takes i (country) and returns all of the x values from each index from each dataset for every year.
        """
        x = []
        for k in range(1, len(self._data)): #Loops through datasets
            for l in range(len(self._indices[k])): #Loops through indexes in datasets
                value = self._data[k].iloc[int(self._meta.iloc[k, 5]) * i  + (self._indices[k][l]),:]
                #                           num of indexes * country num(i) +      which index

                x.append([float(number) for number in value])
        return x
        