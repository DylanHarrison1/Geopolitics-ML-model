from main import Instance
import pandas as pd
import os

#In model, layer is a list. We could pass that all the way in.
Data = [["Demographics", "N R R", "V-Dem"],
        [[6,7,8,93,94,95],[1],[1,2,3,4,5,6,7,8,9,10]]]

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

layerPos = [[20, 20],
            [20, 30, 20],
            [20, 30, 30, 20]]

#every possibility from Data
for i in range(2 ** len(Data[0])):

    #convert i to binary, apply to Data
    binNum = bin(i)[2:]
    boolList = [bit == '1' for bit in binNum]
    newData = [[],[]]
    for j in range(2):
        newData[j] = [Data[j][i] for i, value in enumerate(boolList) if value]

    
    inputSize = 0
    for innerList in newData[1]:
        for item in innerList:
            if isinstance(item, int):
                inputSize += 1
    
    newData[0].insert(0,"HDI")
    newData[1].insert(0,[1])

    for j in range(3):
        accuracy = []
        for k in range(5):
            test = Instance("basic", 
                            [inputSize, (item for item in layerPos[j]), 5],
                            newData[0],
                            newData[1],
                            "slice")
            test.Run(10)
            accuracy.append(test.TestModel())

        mean = 0
        for x in range(5):
            for y in range(5):
                mean += accuracy[y][x]
            mean = mean / 25
        df.iloc[i,2 + j] = mean
    df.to_csv(os.getcwd() + "\Results.csv")    
