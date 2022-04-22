import cv2
import numpy as np


class CamEffect:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)
        if (self.cap.isOpened() == False):
            raise Exception("Error opening video stream or file")

    def capture_frame(self):
        self.ret, self.frame = self.cap.read()
        self.frame = cv2.flip(self.frame, 1)

    def get_edges(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        self.edges = cv2.Canny(blur, 50, 100)
    
    def fizzy_frame(self, mask):
        self.new_mask = np.empty((self.height, self.width), dtype=np.uint64)
        fizzy_range = [-10, 0, 10]
        positions = np.where((mask == 255))
        self.new_mask[np.where((mask == 0))] = 0
        self.new_mask[np.where((mask == 255))] = 0
        for i in range(len(positions[0])):
            new_x = positions[0][i]
            new_y = positions[1][i] + fizzy_range[np.random.randint(0,3)]
            if new_x >= self.width:
                new_x = self.width - 1
            if new_y >= self.height:
                new_y = self.height - 1
            self.new_mask[new_x][new_y] = 255
        print(self.new_mask.shape)


    def black_white(self):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0,0,200])
        upper = np.array([255,255,255])
        self.bnw_mask = cv2.inRange(hsv, lower, upper)

    def colored_bnw(self, fizzy=False):
        self.black_white()
        if fizzy:
            self.fizzy_frame(self.bnw_mask)
        else:
            self.new_mask = self.bnw_mask
        self.frame[np.where((self.new_mask == 0))] = [0, 0, 0]
        self.frame[np.where((self.new_mask == 255))] = self.bg_img[np.where((self.new_mask == 255))]

    def colored_edges(self, fizzy=False):
        self.get_edges()
        if fizzy:
            self.fizzy_frame(self.edges)
        else:
            self.new_mask = self.edges
        self.frame[np.where((self.new_mask == 0))] = [0, 0, 0]
        self.frame[np.where((self.new_mask == 255))] = self.bg_img[np.where((self.new_mask == 255))]
    
    def set_bg_img(self, img_path):
        self.bg_img = cv2.imread(img_path)
        self.capture_frame()
        self.height, self.width, channels = self.frame.shape
        self.bg_img = cv2.resize(self.bg_img, (self.width, self.height), interpolation = cv2.INTER_AREA)

    def run_effect(self, choice="1", fizzy=False):
        self.set_bg_img("yellow.jpg")
        # Read until video is completed
        while(self.cap.isOpened()):
            self.capture_frame()
            if self.ret == True:
                # Display the resulting frame
                if choice == "1":
                    self.colored_edges(fizzy)
                if choice == "2":
                    self.colored_bnw(fizzy)
                cv2.imshow('Frame', self.frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    cam_effect = CamEffect()
    cam_effect.run_effect(choice="1")