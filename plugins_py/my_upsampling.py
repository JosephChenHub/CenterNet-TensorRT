import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.onnx.symbolic_helper import parse_args

class MyUpsample(Function):
    # alpha_f means the `alpha` attribute with type `float`
    # legal types: f(float), i(int), s(string), t(Tensor)
    @staticmethod
    def symbolic(g, input, output_size, scale_factor, align_corners):
        return g.op("MyUpsample", input,
                    namespace_s = "",
                    name_s = "MyUpsample", # plugin name
                    version_s = "v0", # plugin version
                    align_corners_i = align_corners,
                    scale_factor_f = scale_factor,
                    output_size_i = output_size
                    )

    @staticmethod
    def forward(ctx, input, output_size, scale_factor, align_corners):
        if output_size == -1:
            output_size = None
        if scale_factor == -1:
            scale_factor = None
        return F.interpolate(input, size=output_size, scale_factor=scale_factor, mode='bilinear', align_corners=align_corners)

    @staticmethod
    def backward(ctx, grad_output):
        raise Exception("Not implemented!")

class MyUpsampling(nn.Module):
    def __init__(self, output_size=None, scale_factor=None, align_corners=True):
        super().__init__()
        assert output_size is not None or scale_factor is not None
        self.output_size = output_size
        self.scale_factor = scale_factor
        if output_size is None:
            self.output_size = -1
        if scale_factor is None:
            self.scale_factor = -1

        self.align_corners = align_corners

    def set_scale(self, scale_factor):
        assert len(scale_factor) == 2
        self.scale_factor = scale_factor

    def set_size(self, size):
        assert len(size) == 2
        self.output_size = size

    def forward(self, input):
        return  MyUpsample.apply(input, self.output_size, self.scale_factor, self.align_corners)

if __name__ == "__main__":
    model = MyUpsampling(output_size=(64, 64), align_corners=True).cuda()
    x = torch.randn(1, 3, 32, 32).cuda()
    y1 = model(x)
    y2 = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
    print("Diff:", torch.abs(y1-y2).sum())
    print("Test on converting to onnx ...")
    torch.onnx.export(model, x, "toy.onnx", verbose = True, opset_version=11)


    model = nn.Sequential(
        MyUpsampling(scale_factor=(20, 20), align_corners=True),
        nn.Conv2d(3, 3, 3, 1),
        nn.BatchNorm2d(3),
        nn.ReLU(),
    ).cuda()

    model = MyUpsampling(scale_factor=(2.5, 5.5), align_corners=True).cuda()
    x = torch.randn(1, 3, 224, 224, dtype=torch.float).clamp(0.)
    x = x.cuda()
    model.set_scale((5, 10)) # dynamic change
    y1 = model(x)
    y2 = F.interpolate(x, scale_factor=(20, 20), mode='bilinear', align_corners=True)
    print("Test on converting to onnx ...")
    torch.onnx.export(model, x, "toy.onnx", verbose = True, opset_version=11, output_names=['output'])




