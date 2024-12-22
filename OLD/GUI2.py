import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Main Layout
        self.MainLayout = QVBoxLayout()

        # Horizontal Layout for Frames
        self.FrameLayout = QHBoxLayout()

        # Live Camera Feed Label
        self.FeedLabel = QLabel()
        self.FrameLayout.addWidget(self.FeedLabel)

        # Mask Display Label
        self.MaskLabel = QLabel()
        self.FrameLayout.addWidget(self.MaskLabel)

        # Add the Frame Layout to the Main Layout
        self.MainLayout.addLayout(self.FrameLayout)

        # Sliders for HSV Adjustment
        self.HueSlider = QSlider(Qt.Horizontal)
        self.HueSlider.setRange(0, 179)
        self.HueSlider.setValue(0)
        self.HueSlider.setTickInterval(1)
        self.HueSlider.setTickPosition(QSlider.TicksBelow)
        self.HueSlider.setToolTip("Adjust Hue")
        self.MainLayout.addWidget(QLabel("Hue"))
        self.MainLayout.addWidget(self.HueSlider)

        self.SaturationSlider = QSlider(Qt.Horizontal)
        self.SaturationSlider.setRange(0, 255)
        self.SaturationSlider.setValue(0)
        self.SaturationSlider.setTickInterval(1)
        self.SaturationSlider.setTickPosition(QSlider.TicksBelow)
        self.SaturationSlider.setToolTip("Adjust Saturation")
        self.MainLayout.addWidget(QLabel("Saturation"))
        self.MainLayout.addWidget(self.SaturationSlider)

        self.ValueSlider = QSlider(Qt.Horizontal)
        self.ValueSlider.setRange(0, 255)
        self.ValueSlider.setValue(0)
        self.ValueSlider.setTickInterval(1)
        self.ValueSlider.setTickPosition(QSlider.TicksBelow)
        self.ValueSlider.setToolTip("Adjust Value")
        self.MainLayout.addWidget(QLabel("Value"))
        self.MainLayout.addWidget(self.ValueSlider)

        # Cancel Button
        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.MainLayout.addWidget(self.CancelBTN)

        # Worker Thread for Camera Feed
        self.Worker1 = Worker1()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker1.start()

        # Connect sliders to update HSV values
        self.HueSlider.valueChanged.connect(self.UpdateHSV)
        self.SaturationSlider.valueChanged.connect(self.UpdateHSV)
        self.ValueSlider.valueChanged.connect(self.UpdateHSV)

        self.Hue = 0
        self.Saturation = 0
        self.Value = 0

        self.setLayout(self.MainLayout)

    def ImageUpdateSlot(self, ImageTuple):
        Image, Mask = ImageTuple
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))
        self.MaskLabel.setPixmap(QPixmap.fromImage(Mask))

    def UpdateHSV(self):
        self.Hue = self.HueSlider.value()
        self.Saturation = self.SaturationSlider.value()
        self.Value = self.ValueSlider.value()
        self.Worker1.update_hsv(self.Hue, self.Saturation, self.Value)

    def CancelFeed(self):
        self.Worker1.stop()


class Worker1(QThread):
    ImageUpdate = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        self.Hue = 0
        self.Saturation = 0
        self.Value = 0

    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)

        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                # Convert to HSV
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # HSV Mask
                lower_bound = np.array([self.Hue - 10, self.Saturation - 50, self.Value - 50])
                upper_bound = np.array([self.Hue + 10, self.Saturation + 50, self.Value + 50])
                lower_bound = np.clip(lower_bound, 0, 255)
                upper_bound = np.clip(upper_bound, 0, 255)

                mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                # Convert images for PyQt display
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ImageQt = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)

                MaskQt = QImage(mask_bgr.data, mask_bgr.shape[1], mask_bgr.shape[0], QImage.Format_RGB888)

                # Emit updated images
                self.ImageUpdate.emit((ImageQt, MaskQt))

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
