import sys

print(sys.executable)
try:
    import numpy
    print("numpy", numpy.__version__)
except Exception as exc:
    print("numpy_error", repr(exc))

try:
    import torch
    print("torch", torch.__version__)
    print("cuda", torch.cuda.is_available())
except Exception as exc:
    print("torch_error", repr(exc))
