import os
import onnx
import onnxruntime
import torch.onnx
import torch.nn as nn
import argparse
import numpy as np
from train_eval import Net

opts = None

# 将张量转化为ndarray格式
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def init_args():
    global opts
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='', help='source file path', required=True)
    parser.add_argument('--dst', type=str, default='', help='output onnx file store path', required=True)
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    opts = parser.parse_args()

def main():
    dst_model = os.path.join(opts.dst, os.path.basename(os.path.splitext(opts.src)[0])+'.onnx')
    print(dst_model)
    model = torch.load(opts.src)
    print(f"type of loaded model: {type(model)}")
    model.eval()
    dummy_input = torch.randn(opts.batch_size, 1, 28, 28, requires_grad=True)
    torch_out = model(dummy_input)
    torch.onnx.export(model,        # 模型的名称
        dummy_input,   # 一组实例化输入
        dst_model,   # 文件保存路径/名称
        # verbose=True,
        export_params=True,        #  如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.
        opset_version=10,          # ONNX 算子集的版本，当前已更新到15
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names = ['input'],   # 输入模型的张量的名称
        output_names = ['output'], # 输出模型的张量的名称
        # dynamic_axes将batch_size的维度指定为动态，
        # 后续进行推理的数据可以与导出的dummy_input的batch_size不同
        dynamic_axes={'input' : {0 : 'batch_size'},    
                    'output' : {0 : 'batch_size'}})
    
    # 我们可以使用异常处理的方法进行检验
    try:
        # 当我们的模型不可用时，将会报出异常
        onnx.checker.check_model(dst_model)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s"%e)
    else:
        # 模型可用时，将不会报出异常，并会输出“The model is valid!”
        print("The model is valid!")

    ort_session = onnxruntime.InferenceSession(dst_model)
    # 构建输入的字典和计算输出结果
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"torch_out shape: {torch_out.shape} ort_outs shape: {np.array(ort_outs).shape}")
    # 比较使用PyTorch和ONNX Runtime得出的精度
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == '__main__':
    init_args()
    main()
