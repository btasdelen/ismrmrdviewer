
import logging

from PySide6 import QtWidgets, QtCore
from PySide6 import QtWidgets as QTW
from PySide6.QtCore import Qt

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qtagg import FigureCanvas
from .AcquisitionViewer import AcquisitionTable

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from .utils import CachedDataset

# RR: example waveform headers are not arrays
waveform_header_fields = [
    ('version', 'Version', "ISMRMRD Version"),
    ('flags', 'Flags', "Waveform flags bitfield."),
    ('measurement_uid', 'UID', "Unique ID for the measurement."),
    ('scan_counter', 'Scan Counter', "Current waveform number in the measurement."),
    ('time_stamp', 'Waveform Timestamp', "Waveform Timestamp"),
    ('number_of_samples', 'Samples', "Number of samples."),
    ('channels', 'Number of Channels', "Number of channels."),
    ('sample_time_us', 'Sample Time', "Time between samples (in microseconds)"),
    ('waveform_id', 'Waveform ID', "Waveform ID.")
]

class WaveformModel(QtCore.QAbstractTableModel):

    def __init__(self, container):
        super().__init__()

        self.container = container
        self.waveforms = CachedDataset(container.waveforms)

        logging.info("Waveform constructor.")


    def rowCount(self, _=None):
        return len(self.waveforms)

    def columnCount(self, _=None):
        return len(waveform_header_fields)

    def headerData(self, section, orientation, role=Qt.DisplayRole):

        if orientation == Qt.Orientation.Vertical:
            return None

        _, header, tooltip = waveform_header_fields[section]

        if role == Qt.DisplayRole:
            return header
        if role == Qt.ToolTipRole:
            return tooltip

        return None

    def data(self, index, role=Qt.DisplayRole):
        waveform = self.waveforms[index.row()]
        attribute, _, tooltip = waveform_header_fields[index.column()]

        if role == Qt.DisplayRole:
            return getattr(waveform,attribute)
        if role == Qt.ToolTipRole:
            return tooltip

        return None


class WaveformControlGUI(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QHBoxLayout()
        self.channel_selector = QtWidgets.QComboBox()
        self.__set_num_channels(1)
        layout.addWidget(self.channel_selector)

        self.setLayout(layout)

    def __set_num_channels(self, num_channels):
        for i in range(self.channel_selector.count()):
            self.channel_selector.removeItem(i)

        for idx in range(num_channels):
            self.channel_selector.addItem("Channel " + str(idx), userData={"selector": lambda x, i=idx : x[:, i:i + 1],
                                                                           "labeler": lambda scan, coil: str(scan)})

        self.channel_selector.addItem("All Channels", userData={"selector": lambda x: x,
                                                                "labeler": lambda scan, coil: str((scan, coil))})

    def label(self, scan, coil):
        return self.channel_selector.currentData()["labeler"](scan, coil)

    def transform_waveform(self, wave):
        return self.channel_selector.currentData()["selector"](wave)




class WaveformPlotter(FigureCanvas):

    def __init__(self):

        self.figure = mpl.figure.Figure()
        self.axis = np.atleast_1d(self.figure.subplots(1, 1, sharex='col'))
        self.figure.subplots_adjust(hspace=0)

        self.legend = mpl.legend.Legend(self.figure, [], [])
        self.figure.legends.append(self.legend)
        super().__init__(self.figure)

    def clear(self):
        for ax in self.axis:
            ax.clear()

    def plot(self, waveforms,  formatter, labeler):

        for waveform in waveforms:
            x_step = waveform.sample_time_us
            x_scale = np.arange(0, waveform.data.shape[1] * x_step, x_step)
            wave_data = formatter(waveform.data.T)
            for chan, wave in enumerate(wave_data.T):
                self.axis[0].plot(x_scale, wave, label=labeler(waveform.scan_counter,chan))

        handles, labels = self.axis[0].get_legend_handles_labels()
        self.legend = mpl.legend.Legend(self.figure, handles, labels)
        self.figure.legends[0] = self.legend

        self.figure.canvas.draw()

    def plot_concat(self, waveforms,  formatter, labeler):

        self.axis[0].clear()
        wave_data = np.concatenate([formatter(waveform.data.T) for waveform in waveforms])
        x_step = waveforms[0].sample_time_us
        x_scale = np.arange(0, wave_data.shape[0] * x_step, x_step)*1e-6
        self.axis[0].plot(x_scale, wave_data)
        self.axis[0].set_xlabel("Time [s]")

        handles, labels = self.axis[0].get_legend_handles_labels()
        self.legend = mpl.legend.Legend(self.figure, handles, labels)
        self.figure.legends[0] = self.legend

        self.figure.canvas.draw()

    def set_titles(self, titles):
        for ax, title in zip(self.axis, titles):
            ax.set_title(title, loc="right", pad=-10)


class WaveformViewer(QtWidgets.QSplitter):

    def __init__(self, container):
        super().__init__()

        self.model = WaveformModel(container)

        self.waveforms = AcquisitionTable(self)
        self.waveforms.setModel(self.model)
        self.waveforms.setAlternatingRowColors(True)
        self.waveforms.resizeColumnsToContents()
        self.waveforms.selection_changed.connect(self.selection_changed)

        self.setOrientation(Qt.Vertical)

        self.canvas = WaveformPlotter()

        self.bottom_view = QtWidgets.QSplitter()
        self.waveform_gui = WaveformControlGUI()
        self.bottom_view.addWidget(self.waveform_gui)
        self.waveform_gui.channel_selector.currentIndexChanged.connect(self.selection_changed)

        self.concat_btn = QtWidgets.QPushButton("Plot Whole Waveform")
        self.concat_btn.clicked.connect(self.plot_whole_waveform)
        self.bottom_view.addWidget(self.concat_btn)

        self.addWidget(self.waveforms)
        self.addWidget(self.canvas)
        self.addWidget(self.bottom_view)

        self.navigation_toolbar = NavigationToolbar(self.canvas, self.bottom_view)
        self.bottom_view.addWidget(self.navigation_toolbar)

        self.setStretchFactor(0, 6)
        self.setStretchFactor(1, 1)

    def table_clicked(self, index):
        waveform = self.model.waveforms[index.row()]
        self.plot([waveform])

    def format_data(self, acq):
        return self.waveform_gui.transform_waveform(acq.data.T)

    def selection_changed(self):
        self.canvas.clear()

        indices = set([idx.row() for idx in self.waveforms.selectedIndexes()])
        waveforms = [self.model.waveforms[idx] for idx in
                        indices]
        self.canvas.plot(waveforms, self.waveform_gui.transform_waveform, self.waveform_gui.label)

    def plot_whole_waveform(self):
        indices = list([idx.row() for idx in self.waveforms.selectedIndexes()])
        sel_id = self.model.waveforms[indices[-1]].waveform_id
        waveforms = [wf for wf in self.model.waveforms if wf.waveform_id == sel_id]
        self.canvas.plot_concat(waveforms, self.waveform_gui.transform_waveform, self.waveform_gui.label)

