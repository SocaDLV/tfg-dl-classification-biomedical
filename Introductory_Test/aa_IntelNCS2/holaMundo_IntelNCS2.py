import cv2
import numpy as np
import openvino.runtime as ov

core = ov.Core()

model = core.read_model(r"C:\Users\...\saved_model.xml")

compiled_model = core.compile_model(model=model, device_name="CPU") 

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) 
    img = img.astype(np.float32)  
    img = img / 255.0  
    img = img[np.newaxis, :] 
    return img

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)
network_input_shape = input_key.shape

image_path = r"C:\Users\...\val_0.JPEG"
input_image = preprocess_image(image_path)

result = compiled_model(input_image)[output_key]  
result_index = np.argmax(result)

print(result_index)  
