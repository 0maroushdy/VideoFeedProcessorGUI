import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import datetime


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Main Layout ---------------------
        self.MainLayout = QVBoxLayout()

        # Horizontal Layout for Frames ----
        self.FrameLayout = QHBoxLayout()

        # Live Camera Feed Label -----------
        self.FeedLabel = QLabel()
        self.FrameLayout.addWidget(self.FeedLabel)

        # Processed Video Label ---------------
        self.ProcessedLabel = QLabel()
        self.FrameLayout.addWidget(self.ProcessedLabel)

        # Add the Frame Layout to the Main Layout --
        self.MainLayout.addLayout(self.FrameLayout)

        # Sliders for HSV Adjustment -----------
        self.HueSlider = QSlider(Qt.Horizontal)
        self.HueSlider.setRange(0, 179)
        self.HueSlider.setValue(0) 
        self.HueSlider.setTickInterval(3)
        self.HueSlider.setTickPosition(QSlider.TicksBelow)
        self.HueSlider.setToolTip("Adjust Hue")
        self.MainLayout.addWidget(QLabel("Hue"))
        self.MainLayout.addWidget(self.HueSlider)

        self.SaturationSlider = QSlider(Qt.Horizontal)
        self.SaturationSlider.setRange(0, 255)
        self.SaturationSlider.setValue(255)
        self.SaturationSlider.setTickInterval(10)
        self.SaturationSlider.setTickPosition(QSlider.TicksBelow)
        self.SaturationSlider.setToolTip("Adjust Saturation")
        self.MainLayout.addWidget(QLabel("Saturation"))
        self.MainLayout.addWidget(self.SaturationSlider)

        self.ValueSlider = QSlider(Qt.Horizontal)
        self.ValueSlider.setRange(0, 255)
        self.ValueSlider.setValue(255)
        self.ValueSlider.setTickInterval(5)
        self.ValueSlider.setTickPosition(QSlider.TicksBelow)
        self.ValueSlider.setToolTip("Adjust Value")
        self.MainLayout.addWidget(QLabel("Value"))
        self.MainLayout.addWidget(self.ValueSlider)

        # Cancel Button -------------------------
        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.MainLayout.addWidget(self.CancelBTN)

        # # Capture Button -------------------------
        # self.CaptureBTN = QPushButton("Capture Photo")
        # self.CaptureBTN.clicked.connect(self.CapturePhoto)
        # self.MainLayout.addWidget(self.CaptureBTN)


        # Worker Thread for Camera Feed ---------
        self.Worker1 = Worker1()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker1.start()

        # Connect sliders to update HSV values ---
        self.HueSlider.valueChanged.connect(self.UpdateHSV)
        self.SaturationSlider.valueChanged.connect(self.UpdateHSV)
        self.ValueSlider.valueChanged.connect(self.UpdateHSV)

        self.Hue = 0
        self.Saturation = 255
        self.Value = 255

        self.setLayout(self.MainLayout)

    def ImageUpdateSlot(self, ImageTuple):
        Image, Processed = ImageTuple
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))
        self.ProcessedLabel.setPixmap(QPixmap.fromImage(Processed))

    def UpdateHSV(self):
        self.Hue = self.HueSlider.value()
        self.Saturation = self.SaturationSlider.value()
        self.Value = self.ValueSlider.value()
        self.Worker1.update_hsv(self.Hue, self.Saturation, self.Value)

    def CancelFeed(self):
        self.Worker1.stop()

    # Under Dev. &-&
    # def CapturePhoto():
    #     ret, frame = self.Worker1.capture.read() 
    #     if ret:
    #         timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #         filename = f"Captured_{timestamp}.jpg"
    #         cv2.imwrite(filename, frame)
    #         print(f"Photo saved as {filename}")


class Worker1(QThread):
    ImageUpdate = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        self.Hue = 0
        self.Saturation = 255
        self.Value = 255
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)

        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                # Detect Faces and Eyes ---------------------
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray_frame[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    eyes = self.eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                # Convert to HSV and apply sliders ------------
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv_frame[:, :, 0] = np.clip(hsv_frame[:, :, 0] + self.Hue, 0, 179)
                hsv_frame[:, :, 1] = np.clip(hsv_frame[:, :, 1] * (self.Saturation / 255), 0, 255)
                hsv_frame[:, :, 2] = np.clip(hsv_frame[:, :, 2] * (self.Value / 255), 0, 255)
                processed_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)

                # Convert images for PyQt display --------------
                OriginalQt = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_BGR888)
                ProcessedQt = QImage(processed_frame.data, processed_frame.shape[1], processed_frame.shape[0], QImage.Format_BGR888)

                # Emit updated images --------------------------
                self.ImageUpdate.emit((OriginalQt, ProcessedQt))

        Capture.release()

    def update_hsv(self, hue, saturation, value):
        self.Hue = hue
        self.Saturation = saturation
        self.Value = value

    def stop(self):
        self.ThreadActive = False
        self.quit()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())
