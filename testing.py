from main import Instance

#In model, layer is a list. We could pass that all the way in.
Data = [["HDI","Demographics","N R R", "V-Dem"],
        [[1],[],[1],[1,2,3,4,5,6,7,8,9,10]]]


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
test1 = Instance(False, False)
score = []
for i in range(10):
    x = test1.Run(20)
    score.append(x)

mean = [0,0,0,0,0]
print(score)
for i in range(5):
    for j in range(10):
        mean[i] += score[j, i]
    mean[i] = mean[i] / 10
print(mean)


test2 = Instance("basic", 
                 [4,10.10,5],
                 ["HDI", "Demographics", "Natural Resource Rent"],
                 [[1], [97,98,99],[1]],
                 "slice",
                 False,
                 True)
test2.Run(10)
score = test2.TestModel()
"""