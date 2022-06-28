import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import torch


transforms_test = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor()])

classes = {1: "Ball orange", 0: "90 deg angle rotated", 2: "Ball yellow", 3: "Ball green", 4: "45 deg angle rotated"}


def get_coords(field_img, image, x_size_mm, y_size_mm, model_classificator):

    contours, hierarchy = cv2.findContours(image,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = image.shape

    objects_data = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:
            x, y, w, h = cv2.boundingRect(contour)
            #frames.append((x, y, w, h))
            try:
                object_to_nn = field_img[y - 5: y + h + 10, x - 5  : x + w + 10]
            except:
                object_to_nn = field_img[y: y + h, x: x + w]
            object_to_nn = cv2.cvtColor(object_to_nn, cv2.COLOR_BGR2RGB)
            object_to_nn_image = object_to_nn.copy()
            object_to_nn = Image.fromarray(object_to_nn)
            object_to_nn = transforms_test(object_to_nn).unsqueeze(0)
            
            object_to_nn_image_aa = cv2.cvtColor(object_to_nn_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("NN", object_to_nn_image_aa)
            cv2.imshow("Field", field_img)
            out = model_classificator(object_to_nn)
            _, predicted = torch.max(out.data, 1)
            final_object_type_id = predicted.cpu().numpy()[0]
            print(classes[final_object_type_id]) #Visual check
            
           
            x_center = x + w // 2
            y_center = y + h // 2
            x_mm = x_size_mm * x_center // width
            y_mm = y_size_mm - (y_size_mm * y_center // height)
            #centers.append((x_mm, y_mm))
            data_packet = {"centroid_coord": (x_mm, y_mm), "frame_box": (x, y, w, h), "object_class": final_object_type_id}
            objects_data.append(data_packet)

    return objects_data