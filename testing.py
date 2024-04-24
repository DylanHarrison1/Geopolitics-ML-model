from main import Instance
import pandas as pd
import os
import copy

#In model, layer is a list. We could pass that all the way in.


Data = [["V-Party", "Demographics", "Disasters", "Geopol Risk", "N R R", "V-Dem", "Worldcities"],
        [[0,1,2,7,21,22], [6,7,8,93,94,95], [2,3,4,5,6,7,8,9], [0], [0], [1,2,3,4,5,6,7,8,9,10], [0,1,2,3,4]]]

Data2 =  [["V-Party", "Disasters", "Geopol Risk", "N R R", "V-Dem", "Worldcities"],
          [[0,1,2,7,21,22], [2,3,4,5,6,7,8,9], [0], [0], [1,2,3,4,5,6,7,8,9,10], [0,1,2,3,4]]]

df = pd.read_csv(os.getcwd() + "\Results.csv")


"""
test0 = Instance("basic", 
                 [1, 10, 10, 5],
                 ["HDI", "Test"],
                 [[1],[1]],
                 "slice",
                 True,
                 True)
test0.Run(5)
accuracy = test0.TestModel()
print(accuracy)
"""
def BigTest():
    layerPos = [[20, 20],
                [20, 30, 20],
                [20, 30, 30, 20]]

    #every possibility from Data (except for 0)
    #for i in range(86, 2 ** len(Data[0])):
    for i in range(64, 65):
        print("__________________________")

        #convert i to binary, apply to Data
        binNum = bin(i)[2:]
        boolList = [bit == '1' for bit in binNum]
        while (len(boolList) < 7):
            boolList.insert(0, False)
        newData = [[],[]]
        for j in range(2):
            newData[j] = copy.deepcopy([Data[j][i] for i, value in enumerate(boolList) if value])
        
        inputSize = 0
        for innerList in newData[1]:
            for item in innerList:
                if isinstance(item, int):
                    inputSize += 1
        
        print(newData[0], inputSize)
        newData[0].insert(0,"HDI")
        newData[1].insert(0,[1])

        #loops through structure types
        for j in range(3):
            accuracy = []

            layers = copy.deepcopy(layerPos[j])
            layers.insert(0, inputSize)
            layers.append(5)

            #mean of 5
            for k in range(5):
                test = Instance("basic", 
                                layers,
                                newData[0],
                                newData[1],
                                "slice")
                test.Run(5)
                accuracy.append(test.TestModel())

            mean = 0
            for x in range(5):
                for y in range(5):
                    mean += accuracy[y][x]
            mean = mean / 25
            mean = float(mean)
            print(str(i) + " " + str(mean) + str(layers))
            df.at[i, str(j + 1)] = mean
        #df.to_csv(os.getcwd() + "\Results.csv")    

def TCNtest():
    inputSize = 0
    for innerList in Data2[1]:
        for item in innerList:
            if isinstance(item, int):
                inputSize += 1

    Data2[0].insert(0,"HDI")
    Data2[1].insert(0,[1])

    accuracy = []
    for k in range(5):
        test = Instance("TCN", 
                        [40, 50, 40, 20, 10, 1],
                        Data2[0],
                        Data2[1],
                        "slice")
        test.Run(5)
        accuracy.append(test.TestModel())

    mean = 0
    for x in range(5):
        for y in range(5):
            mean += accuracy[y][x]
    mean = mean / 25
    mean = float(mean)
    print(str(mean))
    print(accuracy)


TCNtest()