from main import Instance

#In model, layer is a list. We could pass that all the way in.
Data = [["HDI","Demographics","N R R", "V-Dem"],
        [[1],[6,7,8,93,94,95],[1],[1,2,3,4,5,6,7,8,9,10]]]


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

