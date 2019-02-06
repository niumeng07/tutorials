import torch
import pdb

x = torch.ones(2, 2, requires_grad=True) #will calcute grad when backward() called.
print("x = ", x)

y = x + 2  #requires_grad will be False
y = y.requires_grad_(True)
print("y.requires_grad = ", y.requires_grad)

print("y = ", y)
print("y.grad_fn = ", y.grad_fn)

z = y * y * 3  #requires_grad will be False
z = z.requires_grad_(True)
print("z.requires_grad = ", z.requires_grad)
out = z.mean()  #requires_grad will be False
print("z = ", z, "\tout = ", out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))  # all element-wise
# print(a.requires_grad)

a.requires_grad_(True)
print("a.requires_grad = ", a.requires_grad)

b = (a * a).sum()
#print(b.grad_fn)

print("out = ", out)
out.backward()
print("x.grad = ", x.grad, "\tz.grad = ", z.grad, "\ty.grad = ", y.grad)

print("*" * 100)
x = torch.randn(3, requires_grad=True)
y = x * 2

#while y.data.norm() < 1000:
#    y = y * 2
print("x: ", x)
print("y: ", y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print("x.grad = ", x.grad)
print("x.requires_grad = ", x.requires_grad)
print("(x ** 2).requires_grad = ", (x ** 2).requires_grad)

with torch.no_grad():
  print((x ** 2).requires_grad)

