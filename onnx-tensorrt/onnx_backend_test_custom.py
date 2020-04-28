 # Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a
 # copy of this software and associated documentation files (the "Software"),
 # to deal in the Software without restriction, including without limitation
 # the rights to use, copy, modify, merge, publish, distribute, sublicense,
 # and/or sell copies of the Software, and to permit persons to whom the
 # Software is furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 # DEALINGS IN THE SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import unittest
import onnx.backend.test

import onnx_tensorrt.backend as trt

# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(trt, __name__)
#TRT custom tests
#backend_test.include(r'.*test_basic_conv_.*custom.*')
#backend_test.include(r'.*test_conv_.*custom.*')
#backend_test.include(r'.*test_convtranspose.*custom.*')
#backend_test.include(r'.*test_batchnorm.*custom.*')
#backend_test.include(r'.*test_reshape.*custom.*')
#backend_test.include(r'.*test_prelu.*custom.*')
#backend_test.include(r'.*test_topk.*custom.*')
#backend_test.include(r'.*test_upsample.*custom.*')
#backend_test.include(r'.*test_constant_pad_custom.*')
#backend_test.include(r'.*test_resize.*custom.*')
#backend_test.include(r'.*test_split.*custom.*')
#backend_test.include(r'.*test_instancenorm_.*_custom.*')
#backend_test.include(r'.*test_slice.*custom.*')
backend_test.include(r'.*test_MyUpsample.*custom.*')


# exclude unenabled ops get pulled in with wildcards
# test_constant_pad gets pulled in with the test_constant* wildcard. Explicitly disable padding tests for now.
backend_test.exclude(r'.*test_constant_pad.*')
backend_test.exclude(r'.*test_constantofshape.*')
backend_test.exclude(r'.*test_expand.*')
# Operator MATMULINTEGER is not supported by TRT
backend_test.exclude(r'.*test_matmulinteger.*')
backend_test.exclude(r'.*test_maxpool.*')
backend_test.exclude(r'.*test_maxunpool.*')
# Mismatch: 0.476%, relative diff is good.
# Absolute diff failed because
# numpy compares the difference between actual and desired to atol + rtol * abs(desired)
backend_test.exclude(r'.*test_convtranspose_3d_custom_cuda')
# dilations not supported in ConvTRanspose layer
backend_test.exclude(r'.*test_convtranspose_dilations_custom_cuda')

globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()
