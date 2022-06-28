import torch
from model import Net
from torchvision import transforms
import cv2
import numpy as np

def load_net(path_to_model, device="cpu", type=0):
    device = torch.device(device)
    if type == 0:
        model = torch.load(path_to_model, map_location=torch.device('cpu'))
    else:
        model = Net()
        model.load_state_dict(torch.load(path_to_model))
    model.eval()
    return model

def inference_frame(img, model, device="cpu"):
    '''
    numpy array 3xWxH -> numpy array 3x224x244 -> binary mask 640x480
    '''
    device = torch.device(device)
    transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



    for_mask = np.zeros((224, 224, 3), 'uint8')
    img = cv2.resize(img, (224, 224))
    img = transform(img).to(device)
    img = img.unsqueeze(0)
    output = model(img)
    output_np = output.cpu().data.numpy().copy()
    output_np = np.argmin(output_np, axis=1)
    mask2 = np.squeeze(output_np[0, :, :])
    color = np.array([255, 255, 255], dtype='uint8')
    masked_img = np.where(mask2[...,None], color, for_mask)
    masked_img = cv2.resize(masked_img, (640, 480))
    kernel = np.ones((5, 5), 'uint8')
    img_bin = cv2.erode(masked_img, kernel, iterations=7)
    img_bin = cv2.dilate(img_bin, kernel, iterations=7)
    return img_bin

def warm_up(model, iterations=3, device="cpu"):
    for _ in range(iterations):
        rnd_data = torch.randn(2, 3, 224, 224, device=torch.device(device))


if __name__ == "__main__":
    temp_model = load_net("models/trained_model.pth", "cuda")
    print(temp_model)
    warm_up(temp_model, device="cuda")
    val_img = cv2.imread("tests/val.jpg")
    mask = inference_frame(val_img, temp_model, "cuda")
    cv2.imshow("Test prediction", mask)
    cv2.waitKey(0)