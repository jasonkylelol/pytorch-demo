import onnxruntime
from torch.utils.data import DataLoader
from train_eval import FMDataset, data_transform
import pandas as pd
import torch
import numpy as np

model_path = 'onnx/fashion_mnist_demo.onnx'
batch_size = 1
# workers = 4

def onnx_infer():
    test_df = pd.read_csv('datasets/fashion-mnist_test.csv')
    test_data = FMDataset(test_df, data_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print(f"test_loader length: {len(test_loader)}")
    cnt = 0
    for data, label in test_loader:
        cnt += 1
        if cnt > 10:
            break
        ort_session = onnxruntime.InferenceSession(model_path)
        # 构建输入的字典和计算输出结果
        # print(f"name: {ort_session.get_inputs()[0].name} data shape: {data.shape}")
        ort_inputs = {ort_session.get_inputs()[0].name: data.numpy()}
        output = ort_session.run(None, ort_inputs)
        # print(f"raw output shape: {np.array(output).shape}")

        out_tensor = torch.tensor(np.array(output[0]))
        preds = torch.argmax(out_tensor, 1)
        csv_label = label.numpy()
        predict_label = preds.numpy()

        # print(f"out_tensor shape: {out_tensor.shape}")
        # print(f"label shape: {label.shape} preds shape: {preds.shape}")
        print(f"[{cnt}] csv_label: {csv_label} | predict_label: {predict_label}")

def main():
    onnx_infer()

if __name__ == '__main__':
    main()