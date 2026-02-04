import numpy as np

a = np.array([[2, 5, 6],[1, -2, 4]])
b = np.array([[3,4], [4,-2], [-5,1]])

a = a.reshape(2,3)
print (a.shape,b.shape)
print(a)
print("\n")
print(b)
# b = b.reshape(3,1)
 print(np.dot(a, b))
 print(np.dot(b, a))
