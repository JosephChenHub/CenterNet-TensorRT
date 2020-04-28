
from dcn_v2 import DCN
import torch

x = torch.randn(5, 3, 40, 40).cuda()
net = DCN(x.shape[1], 6, (3,3), 1, 1).cuda()
y = net(x)
print("y.shape:", y.shape)
torch.onnx.export(net, x, 'dcn.onnx', verbose=False, opset_version=9)


x_ = x.view(-1)
x_str =""

for i in range(x_.shape[0]):
    x_str += str(x_[i].item()) + ","
x_str = x_str[:-1]
x_str += "\n"

with open("x_data.txt", "w") as fout:
    fout.writelines(x_str)


x_ = y.view(-1)
x_str =""
for i in range(x_.shape[0]):
    x_str += str(x_[i].item()) + ","
x_str = x_str[:-1]
x_str += "\n"

with open("y_data.txt", "w") as fout:
    fout.writelines(x_str)
