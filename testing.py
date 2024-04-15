from main import Instance

#In model, layer is a list. We could pass that all the way in.

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
                 [[], [97,98,99],[]],
                 "slice",
                 False,
                 True)