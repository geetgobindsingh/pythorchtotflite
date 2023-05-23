import onnx
import onnx_tf
import tensorflow

def onnx2tf():
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)
    # onnx.helper.printable_graph(onnx_model.graph)
    tf_rep = onnx_tf.backend.prepare(onnx_model, device="CPU")
    #tf_rep = prepare(onnx_model)
    tf_rep.export_graph("tf_model")


def tf2tflite():
    converter = tensorflow.lite.TFLiteConverter.from_saved_model("tf_model")
    tflite_model = converter.convert()
    with open("model.tflite", 'wb') as f:
        f.write(tflite_model)



if __name__ == '__main__':
    #onnx2tf()
    tf2tflite()