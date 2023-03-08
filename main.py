import torch
import os
from PytorchNet import PytorchNetV2
# import cv2
import numpy as np
import onnx
import onnx_tf
import tensorflow


def parse_model_name(model_name):
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def load_model(model_path):
    model_name = os.path.basename(model_path)
    print(model_name)
    h_input, w_input, model_type, _ = parse_model_name(model_name)
    device_id = 0
    device = torch.device("cuda:{}".format(device_id)
                          if torch.cuda.is_available() else "cpu")

    kernel_size = get_kernel(h_input, w_input, )
    #model = PytorchNetV1SE(conv6_kernel=kernel_size).to(device)
    model = PytorchNetV2(conv6_kernel=kernel_size).to(device)

    state_dict = torch.load(model_path, map_location=device)
    keys = iter(state_dict)
    first_layer_name = keys.__next__()
    if first_layer_name.find('module.') >= 0:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            name_key = key[7:]
            new_state_dict[name_key] = value
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    return h_input, w_input, model


# Press the green button in the gutter to run the script.
def torch2onnx(torch_model, sample_input):
    torch.onnx.export(
        torch_model,  # PyTorch Model
        sample_input,  # Input tensor
        "model/output_model2.onnx",  # Output file (eg. 'output_model.onnx')
        opset_version=12,  # Operator support version
        input_names=['input'],  # Input tensor name (arbitary)
        output_names = ['output']  # Output tensor name (arbitary)
    )
    return None


def onnx2tf():
    onnx_model = onnx.load("model/output_model2.onnx")
    onnx.checker.check_model(onnx_model)
    # onnx.helper.printable_graph(onnx_model.graph)
    tf_rep = onnx_tf.backend.prepare(onnx_model, device="CPU")
    #tf_rep = prepare(onnx_model)
    tf_rep.export_graph("model/tf_model2")


def tf2tflite():
    converter = tensorflow.lite.TFLiteConverter.from_saved_model("model/tf_model2")
    tflite_model = converter.convert()
    with open("model/model2.tflite", 'wb') as f:
        f.write(tflite_model)

if __name__ == '__main__':
    #model_path = "model/4_0_0_80x80_MiniFASNetV1SE.pth"
    model_path = "model/2.7_80x80_MiniFASNetV2.pth"
    height, width, torch_model = load_model(model_path)
    sample_input = torch.rand((1, 3, height, width))
    torch_model.eval()

    torch2onnx(torch_model, sample_input)
    onnx2tf()
    tf2tflite()

    #torch_output = inference_torch()
