import os
import onnx
import onnxruntime
import torch.onnx
import torch.nn as nn
import argparse
import numpy as np
from train_eval import Net

opts = None


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
    torch.onnx.export(model,        
        dummy_input,   
        dst_model,   
        # verbose=True,
        export_params=True,        
        opset_version=10,         
        do_constant_folding=True, 
        input_names = ['input'],  
        output_names = ['output'],
       
        dynamic_axes={'input' : {0 : 'batch_size'},    
                    'output' : {0 : 'batch_size'}})
    
    
    try:
       
        onnx.checker.check_model(dst_model)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s"%e)
    else:
        
        print("The model is valid!")

    ort_session = onnxruntime.InferenceSession(dst_model)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"torch_out shape: {torch_out.shape} ort_outs shape: {np.array(ort_outs).shape}")

    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == '__main__':
    init_args()
    main()
