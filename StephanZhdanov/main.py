import cv2
import numpy as np
from config import *
from get_bin_neuralnet import load_net
from ui import App
from main_wrapper import recognition
import threading
from robot_control import ManipulatorRobot
from model_classification import CustomConvNet


if __name__ == "__main__":
    net = None
    #outimage = np.zeros((224, 224, 3), 'uint8')
    if DETECTION_TYPE == 1:
        print("Loading model...")
        net = load_net(MODEL_WEIGHTS_PATH, type=0)
    window = App()
    threading.Thread(target=recognition, args=(window, net, )).start()
    window.root.mainloop()

