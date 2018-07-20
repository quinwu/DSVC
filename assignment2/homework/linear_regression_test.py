import numpy as np
import matplotlib.pyplot as plt

#y = 2 * x1 + x2 + 3
#learning rate
rate = 0.001

#train number is 10
x_train = np.array([[1,2],[2,1],[2,3],[3,5],[1,3]])
y_train = np.array([7,8,10,14,8])

#test number is 6
x_test = np.array([[1,4],[2,2],[2,5]])

#create random,init a,b,c
a = 1
b = 1
c = 1


def h(x):
    return a*x[0]+b*x[1]+c             #predict function

for i in range(5000):
    sum_a = 0
    sum_b = 0
    sum_c = 0

    cost_sum = 0

    for x,y in zip(x_train,y_train):
        #gradient descent
        sum_a = sum_a + rate * (y - h(x)) * x[0]
        sum_b = sum_b + rate * (y - h(x)) * x[1]
        sum_c = sum_c + rate * (y - h(x))

        #cost value
        cost_sum = 1/4 * (np.sum((h(x) - y) ** 2))

    #update a,b,c
    a = a + 1/2 * sum_a
    b = b + 1/2 * sum_b
    c = c + 1/2 * sum_c

    print("step %d ------- cost is %f " % (i,cost_sum))
    plt.plot([h(xi) for xi in x_test])

print("%f",a)
print("%f",b)
print("%f",c)

result = [h(xi) for xi in x_train]
print("%f",result)

result = [h(xi) for xi in x_test]
print("%f",result)

plt.show()