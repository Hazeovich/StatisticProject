import torch

step = 0.001
x = torch.tensor(
    [[1.,  2.,  3.,  4.],
     [5.,  6.,  7.,  8.],
     [9., 10., 11., 12.]], requires_grad=True)

#######
device = torch.device('cuda:0' 
                      if torch.cuda.is_available() 
                      else 'cpu')
x = x.to(device)
#######

function = 10 * (x ** 2).sum()
x.retain_grad()
function.backward()

print(x.grad, '<- gradient')
x.data -= step * x.grad
x.grad.zero_()
print(x.grad)
print(x)

step = 0.001
x = torch.tensor([8.,  8.], 
                 requires_grad=True)

optimizer = torch.optim.SGD([x], lr=step)

def function_parabola(var):
    return 10 * (var ** 2).sum()

def make_gradient_step(func, var):
    func_res = func(var)
    var.retain_grad()
    func_res.backward()
    optimizer.step()
    optimizer.zero_grad()

for i in range(500):
    make_gradient_step(function_parabola, x)
    print(x)
    