import tkinter
import PIL.Image, PIL.ImageTk
import cv2

class App:
    def __init__(self):
        self.background_color = "#121212"
        self.foregroud_color = "#FFFFFF"

        self.detection_translations = ["HSV", "NeuralNet"]
        self.root = tkinter.Tk()
        self.root.geometry("600x400")
        self.root.title("Manipulator controller")
        self.root.attributes(
            '-type', 'dialog',
        )
        self.root.configure(background=self.background_color)

        self.label_title = tkinter.Label(self.root, text="General info", font=(None, 25), fg=self.foregroud_color, bg=self.background_color)
        self.label_title.pack()

        self.label_objects_cnt = tkinter.Label(self.root, text="Objects count: N/A", font=(None, 13), fg=self.foregroud_color, bg=self.background_color)
        self.label_objects_cnt.place(x=5, y=40)

        self.label_objects_coords = tkinter.Label(self.root, text="Objects coords: N/A", font=(None, 13), bg=self.background_color, fg=self.foregroud_color)
        self.label_objects_coords.place(x=5, y=70)

        self.label_objects_coords = tkinter.Label(self.root, text="Object id to go for:", font=(None, 13), bg=self.background_color, fg=self.foregroud_color)
        self.label_objects_coords.place(x=5, y=100)

        self.robot_id = tkinter.Entry(self.root, bg=self.background_color, fg=self.foregroud_color)
        self.robot_id.place(x=5, y=130)

        self.send_robot_button = tkinter.Button(self.root, text="Send robot", bg=self.background_color, fg=self.foregroud_color)
        self.send_robot_button.place(x=5, y=160)
    


        


if __name__ == "__main__":
    window = App()
    window.root.mainloop()