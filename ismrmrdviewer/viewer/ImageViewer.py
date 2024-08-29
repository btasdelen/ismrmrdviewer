import logging
import numpy as np
import pdb
import matplotlib.pyplot as pyplot
import matplotlib.animation as animation

from PySide6 import QtCore, QtGui, QtWidgets as QTW

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

DIMS = ('Instance', 'Repetition', 'Set', 'Phase', 'Channel', 'Slice')

class ImageViewer(QTW.QWidget):

    timer_interval = 100 # [ms]
    def __init__(self, container):
        """
        Stores off container for later use; sets up the main panel display
        canvas for plotting into with matplotlib. Also prepares the interface
        for working with multi-dimensional data.
        """
        super().__init__()

        self.container = container
        logging.info("Image constructor.")

        # Main layout
        layout = QTW.QVBoxLayout(self)

        self.nphase = np.max(self.container.images.headers['phase'])+1
        self.nset = np.max(self.container.images.headers['set'])+1
        self.nrep = np.max(self.container.images.headers['repetition'])+1
        self.nimg = int(len(self.container.images)/self.nphase/self.nset/self.nrep) # TODO: Remove this, as it is broken and unnecessary if we handle everything right.

        # Dimension controls; Add a widget with a horizontal layout
        cw = QTW.QWidget()
        layout.addWidget(cw)
        controls = QTW.QHBoxLayout(cw)
        controls.setContentsMargins(0,0,0,0)

        # Create a drop-down for the image instance
        self.dim_buttons = {}
        self.selected = {}
        self.dim_button_grp = QTW.QButtonGroup()
        self.dim_button_grp.setExclusive(True)
        for dim_i, dim in enumerate(DIMS):
            self.dim_buttons[dim] = QTW.QPushButton(text=f'{dim}')
            self.dim_buttons[dim].setCheckable(True)
            self.dim_button_grp.addButton(self.dim_buttons[dim], dim_i)
            controls.addWidget(self.dim_buttons[dim])
            self.selected[dim] = QTW.QSpinBox()
            controls.addWidget(self.selected[dim])
            self.selected[dim].valueChanged.connect(self.update_image)

        self.dim_buttons['Instance'].setChecked(True)
        self.selected['Instance'].setMaximum(self.nimg - 1)
        self.selected['Phase'].setMaximum(self.nphase - 1)
        self.selected['Set'].setMaximum(self.nset - 1)
        self.selected['Repetition'].setMaximum(self.nrep - 1)

        self.animate = QTW.QPushButton()
        self.animate.setCheckable(True)
        pixmapi = QTW.QStyle.StandardPixmap.SP_MediaPlay
        icon = self.style().standardIcon(pixmapi)
        self.animate.setIcon(icon)
        controls.addWidget(self.animate)

        # self.animDim = QTW.QComboBox()
        # for dim in DIMS:
        #     self.animDim.addItem(dim)
        # controls.addWidget(self.animDim)
        controls.addStretch()

        self.animate.clicked.connect(self.animation)
        # self.animDim.currentIndexChanged.connect(self.check_dim)
        self.dim_button_grp.buttonClicked.connect(self.check_dim)

        # Window/level controls; Add a widget with a horizontal layout
        # NOTE: we re-use the local names from above...
        cw = QTW.QWidget()
        layout.addWidget(cw)
        controls = QTW.QHBoxLayout(cw)
        controls.setContentsMargins(0,0,0,0)

        self.windowScaled = QTW.QDoubleSpinBox()
        self.windowScaled.setRange(-2**31, 2**31 - 1)
        self.levelScaled = QTW.QDoubleSpinBox()
        self.levelScaled.setRange(-2**31, 2**31 - 1)
        controls.addWidget(QTW.QLabel("Window:"))
        controls.addWidget(self.windowScaled)
        controls.addWidget(QTW.QLabel("Level:"))
        controls.addWidget(self.levelScaled)
        controls.addStretch()

        self.frameRate = QTW.QDoubleSpinBox()
        self.frameRate.setRange(0.001, 1000)
        self.frameRate.setSuffix(' fps')
        self.frameRate.setValue(10)

        controls.addWidget(QTW.QLabel("Frame Rate:"))
        controls.addWidget(self.frameRate)

        self.frameRate.valueChanged.connect(self.set_timer_interval)

        self.windowScaled.valueChanged.connect(self.window_input)
        self.levelScaled.valueChanged.connect(self.level_input)

        layout.setContentsMargins(0,0,0,0)
        self.fig = Figure(figsize=(6,6),
                          dpi=72,
                          facecolor=(1,1,1),
                          edgecolor=(0,0,0),
                          layout='constrained')

        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.canvas.setSizePolicy(QTW.QSizePolicy.Expanding,
                                  QTW.QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        self.label_base = "A{:d}/S{:d}/C{:d}/P{:d}/R{:d}/S{:d}"
        self.label = QTW.QLabel("")
        self.label.setMaximumSize(140, 20)

        layout.addWidget(self.label)


        data_ = np.flip(np.rot90(np.array(self.container.images.data), axes=(3,4)), axis=4)
        self.stack = np.zeros((self.nimg, self.nrep, self.nset, self.nphase, data_.shape[1], data_.shape[2], data_.shape[3], data_.shape[4]))
        # TODO: We don't have to do the weird trick with the nimg if we properly handle all possible axes.
        # TODO: This indexing does not work properly. Need a better way for handling "unspecified" frames in the header.
        for ii in range(int(data_.shape[0]/self.nimg)):
            for xi in range(self.nimg):
                pi = self.container.images.headers['phase'][ii*xi+ii]
                si = self.container.images.headers['set'][ii*xi+ii]
                ri = self.container.images.headers['repetition'][ii*xi+ii]
                self.stack[xi,ri,si,pi,:,:,:,:] = data_[ii*xi+ii,:,:,:,:]
        # self.stack = np.reshape(self.stack, (self.nimg, self.nset, self.nphase, self.stack.shape[1], self.stack.shape[2], self.stack.shape[3], self.stack.shape[4]))
        if self.stack.shape[0] == 1:
            self.animate.setEnabled(False)

        logging.info("Container size {}".format(str(self.stack.shape)))

        # Window/Level support
        self.min = self.stack.min()
        self.max = self.stack.max()
        self.range = self.max - self.min

        v1 = np.percentile(self.stack,2)
        v2 = np.percentile(self.stack,98)
        self.window = (v2-v1)/self.range
        self.level = (v2+v1)/2/self.range

        self.mloc = None

        # For animation
        self.timer = None

        self.selected['Channel'].setMaximum(self.stack.shape[3]-1)
        self.selected['Slice'].setMaximum(self.stack.shape[4]-1)

        self.update_image()

        for (cont, var) in ((self.windowScaled, self.window),
                            (self.levelScaled, self.level)):
            cont.blockSignals(True)
            cont.setValue(var * self.range)
            cont.blockSignals(False)

    def frame(self):
        "Convenience method"
        return self.selected['Instance'].value()

    def coil(self):
        "Convenience method"
        return self.selected['Channel'].value()

    def slice(self):
        "Convenience method"
        return self.selected['Slice'].value()
    
    def phase(self):
        "Convenience method"
        return self.selected['Phase'].value()
    
    def set(self):
        "Convenience method"
        return self.selected['Set'].value()
    
    def repetition(self):
        return self.selected['Repetition'].value()

    def check_dim(self, v):
        "Disables animation checkbox for singleton dimensions"
        self.animate.setEnabled(self.stack.shape[self.dim_button_grp.checkedId()] > 1)

    def update_wl(self):
        """
        When only window / level have changed, we don't need to call imshow
        again, just update clim.
        """
        rng = self.window_level()
        self.image.set_clim(*rng)        
        self.canvas.draw()

    def window_input(self, value, **kwargs):
        "Handles changes in window spinbox; scales to our [0..1] range"
        self.window = value / self.range 
        self.update_wl()

    def level_input(self, value):
        "Handles changes in level spinbox; scales to our [0..1] range"
        self.level = value / self.range 
        self.update_wl()

    def mouseMoveEvent(self, event):
        "Provides window/level mouse-drag behavior."
        newx = event.position().x()
        newy = event.position().y()
        if self.mloc is None:
            self.mloc = (newx, newy)
            return 
        
        # Modify mapping and polarity as desired
        self.window = self.window - (newx - self.mloc[0]) * 0.01
        self.level = self.level - (newy - self.mloc[1]) * 0.01

        # Don't invert
        if self.window < 0:
            self.window = 0.0
        if self.window > 2:
            self.window = 2.0

        if self.level < 0:
            self.level = 0.0
        if self.level > 1:
            self.level = 1.0

        # We update the displayed (scaled by self.range) values, but
        # we don't want extra update_image calls
        for (cont, var) in ((self.windowScaled, self.window),
                            (self.levelScaled, self.level)):
            cont.blockSignals(True)
            cont.setValue(var * self.range)
            cont.blockSignals(False)

        self.mloc = (newx, newy)
        self.update_wl()

    def mouseReleaseEvent(self, event):
        "Reset .mloc to indicate we are done with one click/drag operation"
        self.mloc = None

    def mouseDoubleClickEvent(self, event):
        v1 = np.percentile(self.stack,2)
        v2 = np.percentile(self.stack,98)
        self.window = (v2-v1)/self.range
        self.level = (v2+v1)/2/self.range
        self.update_wl()

        for (cont, var) in ((self.windowScaled, self.window),
                    (self.levelScaled, self.level)):
            cont.blockSignals(True)
            cont.setValue(var * self.range)
            cont.blockSignals(False)

    def wheelEvent(self, event):
        "Handle scroll event; could use some time-based limiting."
        dimName = self.dim_button_grp.checkedButton().text()
        control = self.selected[dimName]

        num_pixels = event.pixelDelta()
        num_degrees = event.angleDelta() / 8
        delta_steps = 0
        if not num_pixels.isNull():
            delta_steps = num_pixels.y()
        elif not num_degrees.isNull():
            delta_steps = num_degrees.y() / 15

        if delta_steps > 0:
            new_v = control.value() - 1
        elif delta_steps < 0:
            new_v = control.value() + 1
        else:
            return
        control.setValue(max(min(new_v,self.stack.shape[self.dim_button_grp.checkedId()]-1),0))

    def contextMenuEvent(self, event):
    
        menu = QTW.QMenu(self)
        saveAction = menu.addAction("Save Frame")
        transposeAction = menu.addAction("Transpose")

        action = menu.exec(self.mapToGlobal(event.pos()))

        if action == saveAction:
            savefilepath = QTW.QFileDialog.getSaveFileName(self, "Save image as...", filter="Images (*.png, *.jpg, *.svg, *.eps, *.pdf)")
            print(savefilepath)
            if len(savefilepath[0]) != 0:
                extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
                self.fig.savefig(savefilepath[0], bbox_inches=extent)
        elif action == transposeAction:
            self.transpose_image()

    def window_level(self):
        "Perform calculations of (min,max) display range from window/level"
        return (self.level * self.range 
                  - self.window / 2 * self.range + self.min, 
                self.level * self.range
                  + self.window / 2 * self.range + self.min)
    
    def update_image(self, slice_n=None):
        """
        Updates the displayed image when a set of indicies (frame/coil/slice)
        is selected. Connected to singals from the related spinboxes.
        """
        wl = self.window_level()
        self.ax.clear()
        self.image = \
            self.ax.imshow(self.stack[self.frame()][self.repetition()][self.set()][self.phase()][self.coil()][self.slice()], 
                           vmin=wl[0],
                           vmax=wl[1],
                           cmap=pyplot.get_cmap('gray'))
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
        # TODO: Now, this does not make any sense. We could either save the mapping from multidimensional array to linear array we originally started, or look for the idx satifying
        # the header idxs from the headers.
        idx = self.container.images.headers[self.frame()*self.repetition()*self.set()*self.phase()+ self.repetition()*self.set()*self.phase() + self.set()*self.phase()+self.phase()] 
        self.label.setText(self.label_base.format(int(idx['average']),int(idx['slice']),int(idx['contrast']),self.phase(),int(idx['repetition']), self.set()))

    def transpose_image(self):

        self.stack = self.stack.swapaxes(-2,-1)
        self.update_image()

    def set_timer_interval(self, fps):
        self.timer_interval = 1e3/fps

    def animation(self):
        """
        Animation is achieved via a timer that drives the selected animDim
        dimensions' spinbox.
        """
        if self.animate.isChecked() is False:
            if self.timer:
                self.timer.stop()
                self.timer = None
            
            pixmapi = QTW.QStyle.StandardPixmap.SP_MediaPlay
            icon = self.style().standardIcon(pixmapi)
            self.animate.setIcon(icon)
            return
        
        dimName = self.dim_button_grp.checkedButton().text()

        pixmapi = QTW.QStyle.StandardPixmap.SP_MediaPause
        icon = self.style().standardIcon(pixmapi)
        self.animate.setIcon(icon)

        if self.selected[dimName].maximum() == 0:
            logging.warn("Cannot animate singleton dimension.")
            self.animate.setChecked(False)
            return

        def increment():
            "Captures dimName"
            v = self.selected[dimName].value()
            m = self.selected[dimName].maximum()
            self.selected[dimName].setValue((v+1) % m)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.timer_interval)
        self.timer.timeout.connect(increment)
        self.timer.start()

