import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from pyqtgraph import PlotWidget, mkPen, PlotDataItem
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import pyqtgraph as pg
from pyqtgraph import mkPen
from scipy.signal import tf2zpk
from scipy import signal
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer
from datetime import datetime, timedelta
from PyQt5.QtCore import QDateTime
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ui,_=loadUiType('Filter_Creator.ui')


class MainApp(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.actionOpen.triggered.connect(self.upload_signal)
        self.timer_k = QTimer(self)
        self.og_y_vals = []
        self.pad_og_y_vals = []
        self.timer_k.timeout.connect(self.plot_next_point)
        self.UnitCircle = self.findChild(PlotWidget, 'UnitCircle')
        self.Magnitude = self.findChild(PlotWidget, 'Magnitude')
        self.Phase = self.findChild(PlotWidget, 'Phase')
        self.pole = self.findChild(QRadioButton, "poleradio")
        self.zero = self.findChild(QRadioButton, "zeroradio")
        self.conjugate = self.findChild(QCheckBox, "conjugate")
        self.clearzero = self.findChild(QPushButton, "clearzero")
        self.clearpole = self.findChild(QPushButton, "clearpole")
        self.clearall = self.findChild(QPushButton, "clearall")
        self.clearzero.clicked.connect(self.clearzero_clicked)
        self.clearpole.clicked.connect(self.clearpole_clicked)
        self.clearall.clicked.connect(self.clearall_clicked)
        self.allpassbtn.clicked.connect(self.apply_all_pass)
        # Connect radio button signals to a function
        self.pole.toggled.connect(self.radio_button_checked)
        self.zero_flag = True
        self.conjugate_flag = False
        self.conjugate.stateChanged.connect(self.checkbox_checked)
        self.UnitCircle.scene().sigMouseClicked.connect(self.unit_circle_mouse_clicked)
        self.UnitCircle.scene().sigMouseMoved.connect(self.unit_circle_mouse_moved)
        self.clicked_points = []
        self.dragged_point_index = None
        self.plot_unit_circle()
        ######3omda
        self.UnitCircle_2 = self.findChild(PlotWidget, 'UnitCircle_2')
        self.addallpass.clicked.connect(self.handleAddAllPass)
        self.removeallpass.clicked.connect(self.handleRemoveAllPass)
        self.addtolibrary.clicked.connect(self.handleAddToLibrary)
        self.all_pass_library = [-0.9,-0.5,0.0,0.5,0.9,0.5 + 0.5j,1 + 0.5j,1 + 1j,1 + 2j,2 +
                                 0.5j,2 + 2j]
        # contains all the active all pass values
        self.active_library = []
        # Populate the 'Library' ComboBox with items from all_pass_library
        self.Library.addItems([str(item) for item in self.all_pass_library])

        # Pad area mernaaaa
        self.padarea = self.findChild(QLabel, "padarea")
        self.draw_red_line()
        self.amp = []
        self.time_values = []
        self.right_mouse_pressed = False
        self.padarea.resizeEvent = lambda event: self.draw_red_line()
        self.inputsignal = self.findChild(PlotWidget, 'inputsignal')
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        # self.timer.start(100)


    def upload_signal(self):
        self.timer.stop()
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        csv_files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "CSV Files (*.csv);;All Files (*)",
                                                    options=options)
        if csv_files:
            try:
                for csv_file in csv_files:
                    with open(csv_file, 'rb') as file:
                        df = pd.read_csv(csv_file)
                        # get the time and amplitude values from the file
                        y_values = df.iloc[:, -1].values
                        x_values = df.iloc[:, 0]

                        # Store x and y values in instance variables for use in plot_next_point and frequency_response
                        self.x_values = x_values
                        self.y_values = y_values
                        self.current_index = 0

                        # Clear existing plot before starting
                        self.inputsignal.clear()
                        self.filtered_signal.clear()

                        # Start the QTimer with a timeout of 100 milliseconds (adjust as needed)
                        self.timer_k.start(100)
                        print(len(self.x_values))


            except Exception as e:
                print(f'Error reading DAT file: {str(e)}')

    def plot_next_point(self):
        if self.current_index < len(self.x_values):
            print("value : ", self.temporalresslider.value())
            self.timer_k.setInterval(100 - self.temporalresslider.value())
            x = self.x_values[:self.current_index + 1]
            y = self.y_values[:self.current_index + 1]

            # Calculate the frequency response and filtered signal using the current input signal portion
            filtered_signal = signal.lfilter(self.num, self.den, y)
            self.og_y_vals.append(filtered_signal[-1].real)
            self.inputsignal.clear()  # Clear the existing input signal plot
            self.inputsignal.plot(x, y, clear=False)  # Update the input signal plot with new data

            # Plot the filtered signal using the frequency response and filtered_signal
            filtered_x = x
            filtered_y = filtered_signal[:self.current_index + 1].real
            self.filtered_signal.clear()  # Clear the existing filtered signal plot
            self.filtered_signal.plot(filtered_x, np.array(self.og_y_vals),
                                      clear=False)  # Update the filtered signal plot with new data

            self.current_index += 1

            if self.current_index >= 500:
                # Auto-panning: Move the window to the right
                start_index = self.current_index - 500
                end_index = self.current_index
                self.inputsignal.setXRange(self.x_values[start_index], self.x_values[end_index])
                self.filtered_signal.setXRange(self.x_values[start_index], self.x_values[end_index])
        else:
            # Stop the QTimer when all points have been plotted
            self.timer_k.stop()
    def draw_red_line(self):
        pixmap = QPixmap(self.padarea.size())
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        pen = QPen(QColor("red"))
        painter.setPen(pen)
        y = self.padarea.height() // 2
        painter.drawLine(0, y,self.padarea.width(), y)
        painter.end()
        self.padarea.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.timer.start(100)
            self.timer.setInterval(100 - self.temporalresslider.value()) #me7tageen nefakar ezay nesara3haa
            self.right_mouse_pressed = True
            # self.timer_k.stop()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.right_mouse_pressed = False
    # Arbitary signallll
    def mouseMoveEvent(self, event):

        if self.right_mouse_pressed:
            pad_pos = self.padarea.mapFromGlobal(event.globalPos())
            if self.padarea.rect().contains(pad_pos):
                delta_y = self.padarea.height() // 2 -pad_pos.y()
                self.amp.append(delta_y)
                self.time_values.append(datetime.now())
                filtered_signal = signal.lfilter(self.num, self.den, self.amp[-4:])
                ind = len(self.amp)
                self.pad_og_y_vals.extend(filtered_signal[-1:].real)
                print("amp size: ", len(self.amp))
                print("fil:", len(filtered_signal))
                print("pad:", len(self.pad_og_y_vals))
                self.timer_k.stop()



    # graph for arbitary signal
    def update_plot(self):
        if self.amp and self.time_values:
            x_values = list(range(1, len(self.time_values) + 1))
            self.inputsignal.clear()
            self.inputsignal.plot(x=x_values, y=self.amp, pen='r', connect='all')
            self.filtered_signal.clear()
            self.filtered_signal.plot(y = self.pad_og_y_vals)

    def handleRemoveAllPass(self):
        selected_item_index = self.Library_2.currentIndex()
        if selected_item_index != -1:
            self.active_library.pop(selected_item_index)
            self.Library_2.clear()
            self.Library_2.addItems([str(item) for item in self.active_library])
            self.update_all_pass(self.UnitCircle_2)

    def handleAddAllPass(self):
        selected_item_index = self.Library.currentIndex()
        if selected_item_index != -1:
            selected_item = self.all_pass_library[selected_item_index]
            self.active_library.append(selected_item)
            self.Library_2.clear()
            self.Library_2.addItems([str(item) for item in self.active_library])
            self.draw_circle(self.UnitCircle_2)
            zero_x, zero_y, pole_x, pole_y = self.get_zero_pole(selected_item)
            self.draw_zero_pole(self.UnitCircle_2, zero_x, zero_y, pole_x, pole_y)

    def handleAddToLibrary(self):
        input_value = self.valueofalpha.text()
        try:
            all_pass_value = complex(input_value)
            # Check if the value matches a valid pole
            if all_pass_value not in self.all_pass_library:
                # Add the value to the library
                self.all_pass_library.append(all_pass_value)
                self.Library.clear()
                self.Library.addItems([str(item) for item in self.all_pass_library])
            else:
                print(f"{all_pass_value} is already in the library.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    def draw_circle(self, graph):
        theta = np.linspace(0, 2 * np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        circle_item = pg.PlotDataItem(x=x, y=y, pen=mkPen('r'))
        graph.addItem(circle_item)

        # Set aspect ratio to ensure the circle looks like a circle
        graph.getPlotItem().setAspectLocked(lock=True, ratio=1)

        graph.setMenuEnabled(False)
        graph.setLimits(xMin=-2, xMax=2, yMin=-2, yMax=2)
        graph.getViewBox().scaleBy((2, 2))
        graph.showGrid(True, True)

    def apply_all_pass(self):
        self.inputsignal.clear()
        self.filtered_signal.clear()
        self.timer.stop()
        self.timer_k.stop()
        self.og_y_vals = []
        self.pad_og_y_vals = []
        self.amp = []

    def get_all_Pass_zero_pole(self):
        self.all_pass_poles = []
        self.all_pass_zeros = []
        for all_pass in self.active_library:
            b = [-all_pass, 1]
            a = [1, -all_pass]
            z, p, k = tf2zpk(b, a)
            self.all_pass_poles.extend(p)
            self.all_pass_zeros.extend(z)
        return self.all_pass_zeros, self.all_pass_poles

    def draw_zero_pole(self, graph, zero_x, zero_y, pole_x, pole_y):
        zero_item = pg.PlotDataItem(x=[zero_x], y=[zero_y], pen=mkPen('b'), symbol='o', symbolBrush='b')
        pole_item = pg.PlotDataItem(x=[pole_x], y=[pole_y], pen=mkPen('y'), symbol='x', symbolBrush='y')
        graph.addItem(zero_item)
        graph.addItem(pole_item)
        self.get_all_Pass_zero_pole()
        # plot the phase and mag response
        phase, freqs = self.get_all_pass_phase_response(self.all_pass_zeros, self.all_pass_poles)
        self.draw_all_pass_phase_response(self.Phase_2, phase, freqs)

    def get_all_pass_phase_response(self, zeros, poles):
        all_pass_poles = poles
        all_pass_zeros = zeros
        phase = []
        freqs = []
        for point in np.linspace(0, np.pi, 100):
            numerator = 0
            denominator = 0
            y = np.sin(point)
            x = np.cos(point)
            freqs.append(point)
            point = complex(x, y)
            for all_pass in all_pass_zeros:
                numerator += self.calculate_phase(all_pass, point)
            for all_pass in all_pass_poles:
                denominator -= self.calculate_phase(all_pass, point)
            phase.append(numerator + denominator)

        # Convert lists to 1D arrays
        freqs = np.array(freqs)
        phase = np.array(phase)
        return phase, freqs
    def get_zero_pole(self, all_pass):
        b = [-all_pass, 1]
        a = [1, -all_pass]
        z, p, k = tf2zpk(b, a)
        zXCoordinate = np.real(z)
        zYCoordinate = np.imag(z)
        pXCoordinate = np.real(p)
        pYCoordinate = np.imag(p)
        zXCoordinate = zXCoordinate.flatten()[0]
        zYCoordinate = zYCoordinate.flatten()[0]
        pXCoordinate = pXCoordinate.flatten()[0]
        pYCoordinate = pYCoordinate.flatten()[0]
        return zXCoordinate, zYCoordinate, pXCoordinate, pYCoordinate

    def update_all_pass(self, graph):
        self.Phase_2.clear()
        graph.clear()
        self.draw_circle(graph)
        for all_pass in self.active_library:
            zero_x, zero_y, pole_x, pole_y = self.get_zero_pole(all_pass)
            self.draw_zero_pole(graph, zero_x, zero_y, pole_x, pole_y)

    def calculate_phase(self, point1, point2):
        return np.arctan2(point1.imag - point2.imag, point1.real - point2.real)
    def draw_all_pass_phase_response(self, graph, phase, freqs):
        graph.clear()
        print(phase ,freqs)
        freqs = freqs.flatten()
        phase = phase.flatten()
        print(phase, freqs)
        graph.plot(freqs, phase, pen=mkPen(color='b'))
        graph.setTitle('Phase Response')
        graph.setLabel('left', 'Phase')
        graph.setLabel('bottom', 'Frequency (radians)')
        graph.showGrid(True, True)

    def plot_unit_circle(self):
            # Create points for the unit circle
            theta = np.linspace(0, 2 * np.pi, 100)
            x = np.cos(theta)
            y = np.sin(theta)
            # Add a unit circle to the plot
            unit_circle = PlotDataItem(x=x, y=y, pen=mkPen('r'))
            self.UnitCircle.addItem(unit_circle)
            self.UnitCircle.setMenuEnabled(False)
            # self.UnitCircle.setLimits(xMin=-2, xMax=2, yMin=-2, yMax=2)
            self.UnitCircle.getViewBox().scaleBy((1, 1))
            self.UnitCircle.showGrid(True, True)
            self.UnitCircle.getViewBox().setAspectLocked()
            self.UnitCircle.getViewBox().setMouseEnabled(x=False, y=False)

    def radio_button_checked(self):
            # Function to check which radio button is checked
            if self.pole.isChecked():
                self.zero_flag=False
            elif self.zero.isChecked():
                self.zero_flag=True

    def checkbox_checked(self, state):
            # Function to check if the checkbox is checked
            if state == Qt.Checked:
                self.conjugate_flag=True
            else:
                self.conjugate_flag = False
                print("Conjugate checkbox is unchecked")

    def unit_circle_mouse_clicked(self, event):
            # Function to handle left-click or right-click event in UnitCircle plot
            if event.button() == Qt.LeftButton:
                pos = event.scenePos()  # Get the position of the mouse click
                view_pos = self.UnitCircle.plotItem.getViewBox().mapSceneToView(pos)
                # Subtract the center coordinates (0, 0)
                relative_x = round(view_pos.x(), 4)
                relative_y = round(view_pos.y(), 4)
                # Check if the clicked position is very close to an existing point
                for i, (existing_x, existing_y, _) in enumerate(self.clicked_points):
                    distance = np.sqrt((existing_x - relative_x) ** 2 + (existing_y - relative_y) ** 2)
                    if distance < 0.05:
                        if event.button() == Qt.LeftButton :
                            self.dragged_point_index = i
                        return

                # Store the coordinates and attribute of the clicked point (default to 'zero')
                self.clicked_points.append((relative_x, relative_y, 'zero' if self.zero_flag else 'pole'))
                if self.conjugate_flag:
                    # Plot the clicked point
                    if self.zero_flag:
                        self.UnitCircle.plot([relative_x], [relative_y], pen=None, symbol='o', symbolSize=10)
                    else:
                        self.UnitCircle.plot([relative_x, relative_x], [relative_y, relative_y],
                                             pen=mkPen('r'), symbol='x')
                    # Plot the conjugate point
                    conjugate_x = relative_x
                    conjugate_y = -relative_y  # Conjugate of y is -y
                    self.clicked_points.append((conjugate_x, conjugate_y, 'zero' if self.zero_flag else 'pole'))
                    if self.zero_flag:
                        self.UnitCircle.plot([conjugate_x], [conjugate_y], pen=None, symbol='o', symbolSize=10)
                    else:
                        self.UnitCircle.plot([conjugate_x, conjugate_x], [conjugate_y, conjugate_y],
                                             pen=mkPen('r'), symbol='x')
                else:
                    # Plot only the clicked point
                    if self.zero_flag:
                        self.UnitCircle.plot([relative_x], [relative_y], pen=None, symbol='o', symbolSize=10)
                    else:
                        self.UnitCircle.plot([relative_x, relative_x], [relative_y, relative_y],
                                             pen=mkPen('r'), symbol='x')
            elif event.button() == Qt.RightButton:
                # Function to handle right-click event in UnitCircle plot
                pos = event.scenePos()  # Get the position of the mouse click
                view_pos = self.UnitCircle.plotItem.getViewBox().mapSceneToView(pos)
                # Subtract the center coordinates (0, 0)
                relative_x = round(view_pos.x(), 4)
                relative_y = round(view_pos.y(), 4)
                # Check if the right-clicked position is very close to an existing point
                for i, (existing_x, existing_y, _) in enumerate(self.clicked_points):
                    distance = np.sqrt((existing_x - relative_x) ** 2 + (existing_y - relative_y) ** 2)
                    if distance < 0.05:  # threshold
                        # Delete the clicked point and its conjugate
                        del self.clicked_points[i]
                        conjugate_x = existing_x
                        conjugate_y = -existing_y  # Conjugate of y is -y
                        if (conjugate_x, conjugate_y, _) in self.clicked_points:
                            self.clicked_points.remove((conjugate_x, conjugate_y, _))
                        break
                self.re_plot()
            self.plot_frequency_response(self.Magnitude, self.Phase)
            print(self.clicked_points)

    def unit_circle_mouse_moved(self, event):
            if self.dragged_point_index is not None and (QGuiApplication.mouseButtons() & Qt.LeftButton):
                pos = event
                view_pos = self.UnitCircle.plotItem.getViewBox().mapSceneToView(pos)
                relative_x = round(view_pos.x(), 4)
                relative_y = round(view_pos.y(), 4)

                # Update the position of the dragged point
                self.clicked_points[self.dragged_point_index] = (
                relative_x, relative_y, self.clicked_points[self.dragged_point_index][2])

                # Clear and re-plot the unit circle
                self.UnitCircle.clear()
                self.plot_unit_circle()

                # Plot all points based on the updated clicked_points
                for x, y, attribute in self.clicked_points:
                    if attribute == 'zero':
                        symbol = 'o'
                    elif attribute == 'pole':
                        symbol = 'x'
                    if self.zero_flag:
                        self.UnitCircle.plot([x], [y], pen=None, symbol=symbol, symbolSize=10)
                    else:
                        self.UnitCircle.plot([x, x], [y, y], pen=mkPen('r'), symbol=symbol)
            else:
                self.dragged_point_index=None

                self.plot_frequency_response(self.Magnitude, self.Phase)

    def clearzero_clicked(self):
            # Function to clear all zero points from the plot and the list
            self.clicked_points = [(x, y, attr) for x, y, attr in self.clicked_points if attr != 'zero']
            self.re_plot()
            self.plot_frequency_response(self.Magnitude, self.Phase)
            print(self.clicked_points)

    def clearpole_clicked(self):
            # Function to clear all pole points from the plot and the list
            self.clicked_points = [(x, y, attr) for x, y, attr in self.clicked_points if attr != 'pole']
            self.re_plot()
            self.plot_frequency_response(self.Magnitude, self.Phase)
            print(self.clicked_points)

    def clearall_clicked(self):
            # Function to clear all points from the plot and the list
            self.clicked_points.clear()
            self.re_plot()
            self.plot_frequency_response(self.Magnitude, self.Phase)
            print(self.clicked_points)

    def re_plot(self):
            self.UnitCircle.clear()
            self.plot_unit_circle()
            for x, y, attribute in self.clicked_points:
                if attribute == "zero":
                    self.UnitCircle.plot([x], [y], pen=None, symbol='o', symbolSize=10)
                else:
                    self.UnitCircle.plot([x, x], [y, y], pen=mkPen('r'), symbol='x')

    def plot_frequency_response(self, magnitude_plot, phase_plot):
            W,FreqRes=self.frequency_response()
            magnitude_plot.clear()
            magnitude_plot.plot(W, 20*np.log10((abs(FreqRes))), pen=mkPen(color='b'))
            magnitude_plot.setTitle('Magnitude Response')
            magnitude_plot.setLabel('left', 'Magnitude')
            magnitude_plot.setLabel('bottom', 'Frequency (radians)')
            magnitude_plot.showGrid(True, True)

            num, den = signal.zpk2tf(self.all_zeros, self.all_poles, 1)
            allw, AllFreqRes = signal.freqz(num, den)
            phase_plot.clear()
            phase_plot.plot(allw, np.angle(AllFreqRes,deg=True), pen=mkPen(color='b'))
            phase_plot.setTitle('Phase Response')
            phase_plot.setLabel('left', 'Phase')
            phase_plot.setLabel('bottom', 'Frequency (radians)')
            phase_plot.showGrid(True, True)

    def frequency_response(self):
        self.o_zeros = [complex(z[0], z[1]) for z in self.clicked_points if z[2] == 'zero']
        self.o_poles = [complex(p[0], p[1]) for p in self.clicked_points if p[2] == 'pole']
        n_zeros, n_poles = self.get_all_Pass_zero_pole()
        self.all_zeros = np.concatenate((n_zeros, self.o_zeros))
        self.all_poles = np.concatenate((n_poles, self.o_poles))
        # /////////////////////////////////////
        self.num, self.den = signal.zpk2tf(self.o_zeros, self.o_poles, 1)
        w, FreqResp = signal.freqz(self.num, self.den)
        return w, FreqResp
def main():
    app = QApplication(sys.argv)
    QApplication.processEvents()
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()