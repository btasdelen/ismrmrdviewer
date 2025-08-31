import logging
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')

from PySide6 import QtCore, QtWidgets as QTW
from PySide6.QtGui import QIcon

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import animation
from functools import cache
import ismrmrd

DIMS = ('Instance', 'Repetition', 'Set', 'Phase', 'Channel', 'Slice', 'Contrast')

class ImageViewer(QTW.QWidget):

    timer_interval = 100 # [ms]
    def __init__(self, container: ismrmrd.file.Container):
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

        self.nphase = int(np.max(self.container.images.headers['phase'])+1)
        self.nset = int(np.max(self.container.images.headers['set'])+1)
        self.nrep = int(np.max(self.container.images.headers['repetition'])+1)
        self.nslice = int(np.max(self.container.images.headers['slice'])+1)
        self.ncontrast = int(np.max(self.container.images.headers['contrast'])+1)

        self.nimg = int(len(self.container.images)/self.nphase/self.nset/self.nrep/self.nslice/self.ncontrast) # TODO: Remove this, as it is broken and unnecessary if we handle everything right.

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
        self.selected['Slice'].setMaximum(self.nslice - 1)
        self.selected['Contrast'].setMaximum(self.ncontrast - 1)

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

        self.animate.clicked.connect(self.animate_frames)
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

        # Image rotation/flip controls
        # TODO: What if we don't have these icons on the system? Need local fallback icons. Maybe from arrShow project?
        icon_rot_ccw = QIcon.fromTheme("object-rotate-left")
        icon_rot_cw = QIcon.fromTheme("object-rotate-right")
        icon_flip_h = QIcon.fromTheme("object-flip-horizontal")
        icon_flip_v = QIcon.fromTheme("object-flip-vertical")

        self.rot_cw_button = QTW.QPushButton()
        self.rot_cw_button.setIcon(icon_rot_cw)
        self.rot_cw_button.setToolTip("Rotate clockwise")
        self.rot_ccw_button = QTW.QPushButton()
        self.rot_ccw_button.setIcon(icon_rot_ccw)
        self.rot_ccw_button.setToolTip("Rotate counter-clockwise")
        self.flip_h_button = QTW.QPushButton()
        self.flip_h_button.setIcon(icon_flip_h)
        self.flip_h_button.setToolTip("Flip horizontally")
        self.flip_v_button = QTW.QPushButton()
        self.flip_v_button.setIcon(icon_flip_v)
        self.flip_v_button.setToolTip("Flip vertically")

        self.nrot_ = 0
        self.fliph_ = False
        self.flipv_ = False

        self.rot_cw_button.clicked.connect(self.rotate_cw)
        self.rot_ccw_button.clicked.connect(self.rotate_ccw)
        self.flip_h_button.clicked.connect(self.flip_h)
        self.flip_v_button.clicked.connect(self.flip_v)

        controls.addWidget(self.rot_cw_button)
        controls.addWidget(self.rot_ccw_button)
        controls.addWidget(self.flip_h_button)
        controls.addWidget(self.flip_v_button)

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

        try:
            # TODO: This is ugly.
            container.images.data.dtype['imag'] # Test if complex
            data_ = np.flip(np.rot90(np.abs(container.images.data[:,:,:,:,:]['real'] + 1j*container.images.data[:,:,:,:,:]['imag']), axes=(3,4)), axis=4)
        except KeyError as e:
            data_ = np.flip(np.rot90(container.images.data, axes=(3,4)), axis=4)
       
        self.data_ = data_
        # TODO: This indexing does not work properly. Need a better way for handling "unspecified" frames in the header.

        if self.nimg == 1:
            self.animate.setEnabled(False)

        # logging.info("Container size {}".format(str(self.stack.shape)))
        logging.info("Container size {}".format(str(self.image_shape())))

        # Window/Level support
        self.min = self.current_frame().min()
        self.max = self.current_frame().max()
        self.range = self.max - self.min

        v1 = np.percentile(self.current_frame(),2)
        v2 = np.percentile(self.current_frame(),98)
        self.window = (v2-v1)/self.range
        self.level = (v2+v1)/2/self.range

        self.mloc = None

        # For animation
        self.timer = None

        self.selected['Channel'].setMaximum(self.fetch_image(self.repetition(), self.set(), self.phase(), self.slice(), self.contrast()).shape[1]-1)
        # self.selected['Slice'].setMaximum(self.fetch_image(self.repetition(), self.set(), self.phase()).shape[2]-1)

        # Disable widgets for singleton dimensions
        for dim_i, dim in enumerate(DIMS):
            if self.selected[dim].maximum() == 0:
                self.dim_buttons[dim].setEnabled(False)
                self.selected[dim].setEnabled(False)
                self.dim_buttons[dim].setVisible(False)
                self.selected[dim].setVisible(False)

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
    
    def contrast(self):
        "Convenience method"
        return self.selected['Contrast'].value()
    
    def repetition(self):
        return self.selected['Repetition'].value()

    def image_shape(self):
        return (self.nimg, self.nrep, self.nset, self.nphase, self.nslice, self.ncontrast,
                self.fetch_image(self.repetition(), self.set(), self.phase(), self.slice(), self.contrast()).shape[1],
                self.fetch_image(self.repetition(), self.set(), self.phase(), self.slice(), self.contrast()).shape[2],
                self.fetch_image(self.repetition(), self.set(), self.phase(), self.slice(), self.contrast()).shape[3], 
                self.fetch_image(self.repetition(), self.set(), self.phase(), self.slice(), self.contrast()).shape[4])
    
    def check_dim(self, v):
        "Disables animation checkbox for singleton dimensions"
        checkedId = self.dim_button_grp.checkedId()
        # self.animate.setEnabled(self.stack.shape[checkedId] > 1)
        self.animate.setEnabled(bool(self.image_shape()[checkedId] > 1))

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

    def rotate_cw(self):
        self.nrot_ += 1
        self.nrot_ = self.nrot_ % 4
        self.update_image()
    
    def rotate_ccw(self):
        self.nrot_ -= 1
        self.nrot_ = self.nrot_ % 4
        self.update_image()

    def flip_h(self):
        self.fliph_ = not self.fliph_
        self.update_image()

    def flip_v(self):
        self.flipv_ = not self.flipv_
        self.update_image()

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
        v1 = np.percentile(self.current_frame(),2)
        v2 = np.percentile(self.current_frame(),98)
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
        control.setValue(max(min(new_v,self.image_shape()[self.dim_button_grp.checkedId()]-1),0))
        # control.setValue(max(min(new_v,self.stack.shape[self.dim_button_grp.checkedId()]-1),0))

    def contextMenuEvent(self, event):
    
        menu = QTW.QMenu(self)
        saveAction = menu.addAction("Save Frame")
        saveMovieAction = menu.addAction("Save Movie")
        plotFrameAction = menu.addAction("Plot Frame")
        showMetaAction = menu.addAction("Show Image Meta Data")

        action = menu.exec(self.mapToGlobal(event.pos()))

        if action == saveAction:
            savefilepath = QTW.QFileDialog.getSaveFileName(self, "Save image as...", filter="Images (*.png, *.jpg, *.svg, *.eps, *.pdf);;MAT file (*.mat);;NPY file (*.npy)")
            print(savefilepath)
            sel_filter = savefilepath[1]
            if len(savefilepath[0]) != 0:
                if sel_filter == "Images (*.png, *.jpg, *.svg, *.eps, *.pdf)":
                    extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
                    self.fig.savefig(savefilepath[0], bbox_inches=extent)
                elif sel_filter == "MAT file (*.mat)":
                    spio.savemat(savefilepath[0], {'data': self.fetch_image(self.repetition(), self.set(), self.phase(), self.slice(), self.contrast())[self.frame()][self.coil()][0]})
                elif sel_filter == "NPY file (*.npy)":
                    np.save(savefilepath[0], self.fetch_image(self.repetition(), self.set(), self.phase(), self.slice(), self.contrast())[self.frame()][self.coil()][0])
                    
        elif action == saveMovieAction:
            self.save_movie()

        elif action == plotFrameAction:
            wl = self.window_level()
            plt.figure()
            plt.imshow(self.current_frame(), 
                           vmin=wl[0],
                           vmax=wl[1],
                           cmap=plt.get_cmap('gray'))
            plt.axis('off')
            plt.title(f'Frame REP{self.repetition()}/SET{self.set()}/PHS{self.phase()}/SLC{self.slice()}/CH{self.coil()}')
            # self.canvas.draw()
            plt.draw()
            plt.show(block=False)
        elif action == showMetaAction:
            idx_ = np.nonzero((self.container.images.headers['phase'] == self.phase()) & (self.container.images.headers['set'] == self.set()) & (self.container.images.headers['repetition'] == self.repetition()))[0][0]
            meta = ismrmrd.Meta.deserialize(self.container.images.attributes[idx_])
            def fill_item(item, value):
                item.setExpanded(True)
                if type(value) is dict:
                    for key, val in sorted(value.items()):
                        child = QTW.QTreeWidgetItem()
                        child.setText(0, str(key))
                        item.addChild(child)
                        fill_item(child, val)
                elif type(value) is list:
                    for val in value:
                        child = QTW.QTreeWidgetItem()
                        item.addChild(child)
                        if type(val) is dict:      
                            child.setText(0, '[dict]')
                            fill_item(child, val)
                        elif type(val) is list:
                            child.setText(0, '[list]')
                            fill_item(child, val)
                        else:
                            child.setText(0, str(val))              
                        child.setExpanded(True)
                else:
                    child = QTW.QTreeWidgetItem()
                    child.setText(0, str(value))
                    item.addChild(child)

            def fill_widget(widget, value):
                widget.clear()
                fill_item(widget.invisibleRootItem(), value)

            
            popup = QTW.QDialog(self)
            popup.setWindowTitle(f'Meatadata of frame REP{self.repetition()}/SET{self.set()}/PHS{self.phase()}/SLC{self.slice()}/CH{self.coil()}')
            popup.setLayout(QTW.QVBoxLayout())
            widget = QTW.QTreeWidget()
            popup.layout().addWidget(widget)
            fill_widget(widget, dict(meta))
            popup.exec()

    def window_level(self):
        "Perform calculations of (min,max) display range from window/level"
        return (self.level * self.range 
                  - self.window / 2 * self.range + self.min, 
                self.level * self.range
                  + self.window / 2 * self.range + self.min)
    @cache
    def fetch_image(self, repetition, set, phase, slice, contrast):
        "Fetches the image data for the given indicies"

        idx_ = ((self.container.images.headers['phase'] == phase) & 
                (self.container.images.headers['set'] == set) & 
                (self.container.images.headers['repetition'] == repetition) &
                (self.container.images.headers['slice'] == slice) &
                (self.container.images.headers['contrast'] == contrast))
        return self.data_[idx_,:,:,:,:]
        # return None

    def current_frame(self):
        im_ = self.fetch_image(self.repetition(), self.set(), self.phase(), self.slice(), self.contrast())[self.frame()][self.coil()][0]
        if self.fliph_:
            im_ = np.flip(im_, axis=1)
        if self.flipv_:
            im_ = np.flip(im_, axis=0)
        im_ = np.rot90(im_, k=self.nrot_, axes=(0,1))
        return im_
    
    def save_movie(self):
        mid_ = self.container.images.headers[0]['measurement_uid']
        dset_name = self.container.images.headers.parent.parent.name # Hacky
        dim_name = self.dim_button_grp.checkedButton().text()
        framerate = self.frameRate.value()
        movie_filename = f"movie_MID{mid_}_{dset_name}_{dim_name}_{framerate}fps.mp4"
        movie_filename = movie_filename.replace(" ", "_").replace(":", "_").replace("/", "")

        logging.info(f"Saving the movie from dimension {dim_name} with frame rate {framerate} fps as the filename {movie_filename}")

        fig = plt.figure(frameon=False)
        w,h = self.current_frame().shape[0:2]
        dpi = 96
        w /= dpi
        h /= dpi
        fig.set_dpi(dpi)
        fig.set_size_inches(w,h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im_ax = []
        dim_idxs = [self.repetition(), self.set(), self.phase(), self.slice(), self.contrast()]
        Nframes = self.selected[dim_name].maximum()
        for ii in range(Nframes):
            dim_idxs[self.dim_button_grp.checkedId()-1] = ii
            im_ = self.fetch_image(*dim_idxs)[self.frame()][self.coil()][0]
            if self.fliph_:
                im_ = np.flip(im_, axis=1)
            if self.flipv_:
                im_ = np.flip(im_, axis=0)
            im_ = np.rot90(im_, k=self.nrot_, axes=(0,1))
            ima_ = ax.imshow(im_, cmap='gray', animated=True, vmin=self.window_level()[0], vmax=self.window_level()[1], aspect='equal')
            im_ax.append([ima_])

        ani = animation.ArtistAnimation(fig, im_ax, interval=1e3/framerate, blit=True)
        MWriter = animation.FFMpegWriter(fps=framerate)
        ani.save(movie_filename, writer=MWriter)
        logging.info(f"Movie saved as {movie_filename}")
    
    def update_image(self, slice_n=None):
        """
        Updates the displayed image when a set of indicies (frame/coil/slice)
        is selected. Connected to singals from the related spinboxes.
        """
        wl = self.window_level()
        self.ax.clear()
        self.image = \
            self.ax.imshow(self.current_frame(), 
                            vmin=wl[0],
                            vmax=wl[1],
                            cmap=plt.get_cmap('gray'))
            # self.ax.imshow(self.stack[self.frame()][self.repetition()][self.set()][self.phase()][self.coil()][self.slice()], 
            #                vmin=wl[0],
            #                vmax=wl[1],
            #                cmap=plt.get_cmap('gray'))

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
        # TODO: Now, this does not make any sense. We could either save the mapping from multidimensional array to linear array we originally started, or look for the idx satifying
        # the header idxs from the headers.
        idx = self.container.images.headers[self.frame()*self.repetition()*self.set()*self.phase()+ self.repetition()*self.set()*self.phase() + self.set()*self.phase()+self.phase()] 
        self.label.setText(self.label_base.format(int(idx['average']),int(idx['slice']),int(idx['contrast']),self.phase(),int(idx['repetition']), self.set()))

    def transpose_image(self):
        # TODO
        # self.stack = self.stack.swapaxes(-2,-1)
        self.update_image()

    def set_timer_interval(self, fps):
        self.timer_interval = 1e3/fps

    def animate_frames(self):
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

