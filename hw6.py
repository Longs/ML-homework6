import numpy as np 
def hinge_loss_grad(x, y, a):
    mult = 0 if y*a > 0 else 1
    return mult*x*np.sign(y)
    

x = np.transpose([np.array([0,1,2])])
y = 1
a = -1
print(x)
print(hinge_loss_grad(x,y,a))


def SM(z):
    return np.exp(z) / np.sum(np.exp(z))
    
w = np.array([[1, -1, -2], [-1, 2, 1]])
x = np.array([[1], [1]])
y = np.array([[0, 1, 0]]).T
z = np.dot(w.T, x)
a = SM(z)
print(a)
g = np.dot(x, (a - y).T)
print(g.tolist())
new = w-0.5*g
print("\n")
print(new.tolist())
print("\n")
z = np.dot(new.T, x)
a = SM(z)
print(a)