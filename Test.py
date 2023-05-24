import tensorflow as tf
import cv2
import keras.preprocessing.image as image
import numpy as np
from util import transform as trans

def detect_face(img, faceCascade):
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(110, 110)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    return faces


def test(interpreter, img):
    interpreter.allocate_tensors()
    input = interpreter.tensor(interpreter.get_input_details()[0]["index"])
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
    #input(), output()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    return predictions


def display_image(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)


def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def print_inputs_torch(resized_image):
    test_transform = trans.Compose([
        trans.ToTensor(),
    ])
    img = test_transform(resized_image)
    # print("print_inputs_torch test_transform", img.shape)
    img = img.unsqueeze(0).to("cpu")
    # print("print_inputs_torch unsqueeze", img.shape)
    return img


def print_inputs_tflite(resized_image):
    img = image.img_to_array(resized_image, "channels_first")
    # print("print_inputs_tflite img_to_array", img.shape)
    img = np.expand_dims(img, axis=0)
    # print("print_inputs_tflite expand_dims", img.shape)
    return img


def open_camera_test(interpreter, faceCascade):
    # # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening camera")
        exit(0)

    width = 320
    height = 240
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    while True:
        ret, img_bgr = cap.read()
        if ret is False:
            print("Error grabbing frame from camera")
            break

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        faces = detect_face(img_gray, faceCascade)

        for i, (x, y, w, h) in enumerate(faces):

            roi = img_bgr[y:y + h, x:x + w]

            resized_image = cv2.resize(roi, (80, 80))

            img = print_inputs_tflite(resized_image)

            prediction = test(interpreter, img)
            label = np.argmax(prediction)
            score = prediction[label]
            point = (x, y - 5)
            if label == 1:
                print("Real", score)
                text = "Real"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img_bgr, text=text, org=point, fontFace=font, fontScale=0.9,
                            color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            else:
                print("Fake", score)
                text = "Fake"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img_bgr, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
                            thickness=2, lineType=cv2.LINE_AA)

            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)


        cv2.imshow('img_rgb', img_bgr)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def check_image_now(interpreter, input_image):
    img = cv2.imread(input_image)
    img = cv2.resize(img, (int(img.shape[0] * 3 / 4), img.shape[0]))
    result = check_image(img)

    # display_image(img, "Fake")
    faces = detect_face(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), faceCascade)

    for i, (x, y, w, h) in enumerate(faces):
        roi = img[y:y + h, x:x + w]
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        resized_image = cv2.resize(roi, (80, 80))
        # display_image(resized_image, jn"Fake")

        img = print_inputs_tflite(resized_image)
        prediction = test(interpreter, img)
        label = np.argmax(prediction)
        score = prediction[label]
        if label == 1:
            print(input_image, "Real", score)
        else:
            print(input_image, "Fake", score)


if __name__ == '__main__':
    cascPath = "model/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    TF_LITE_MODEL_FILE_NAME = "model/model2.tflite"
    interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print("Input Details:", input_details)
    # print("Input Shape:", input_details[0]['shape'])
    # print("Input Type:", input_details[0]['dtype'])
    #print("Output Shape:", output_details[0]['shape'])
    #print("Output Type:", output_details[0]['dtype'])
    check_image_now(interpreter, "sample/image_F1.jpg")
    check_image_now(interpreter, "sample/image_F2.jpg")
    check_image_now(interpreter, "sample/image_T1.jpg")
    # open_camera_test(interpreter, faceCascade)
        #print_inputs_torch(resized_image)

        # print("img", img)

    #test(interpreter)

