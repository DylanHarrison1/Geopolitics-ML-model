from main import Instance
import pandas as pd
import os
import copy

#In model, layer is a list. We could pass that all the way in.


Data = [["Demographics", "Disasters" "Geopol Risk" "N R R", "V-Dem", "WorldCities"],
        [[6,7,8,93,94,95], [2,3,4,5,6,7,8,9], [0], [0],[1,2,3,4,5,6,7,8,9,10], [0,1,2,3,4]]]

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

#every possibility from Data (except for 0)
for i in range(1, 2 ** len(Data[0])):
    print("__________________________")

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

    #loops through structure types
    for j in range(3):
        accuracy = []

        layers = copy.deepcopy(layerPos[j])
        layers.insert(0, inputSize)
        layers.append(5)
        print(layers)

        #mean of 5
        for k in range(5):
            test = Instance("basic", 
                            layers,
                            newData[0],
                            newData[1],
                            "slice")
            #test.Run(5)
            #accuracy.append(test.TestModel())
            accuracy.append([2,2,2,2,2])

        mean = 0
        for x in range(5):
            for y in range(5):
                mean += accuracy[y][x]
        mean = mean / 25
        mean = float(mean)
        print(str(i) + " " + str(mean))
        df.iat[2 + j, i] = mean
    df.to_csv(os.getcwd() + "\Results.csv")    
