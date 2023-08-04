import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtSvg import QSvgWidget
from RDPMSpecIdentifier.visualize.dashboard import gui_wrapper
import multiprocessing
import os
import time


FPATH = os.path.abspath(__file__)
BASE = os.path.dirname(FPATH)


STYLESHEET = os.path.join(BASE, "style.css")
with open(STYLESHEET) as handle:
    STYLE = handle.read()

ICON = os.path.join(BASE, "Icon.svg")
HEADER = os.path.join(BASE, "RDPMSpecIdentifier_dark_no_text.svg")
assert os.path.exists(HEADER), f"{HEADER} File does not exist"
assert os.path.exists(ICON), f"{ICON} File does not exist"


class DragAndDropWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(50)

        self.text = QLabel("Drag and Drop File")
        self.text.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.text)
        self.setLayout(layout)


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.setText(files[0])


    def setText(self, text):
        if text == "":
            text = "Drag and Drop File"
        self.text.setText(text)


class HeaderLine(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        svg = QSvgWidget(HEADER)
        svg.setMaximumWidth(300)
        svg.setMinimumSize(300, 50)
        svg.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        layout.addWidget(svg)
        self.setLayout(layout)


class DashRunner(QThread):
    def __init__(self, intensities, design, host, port, logbase, sep, text_output):
        super().__init__()
        self.intensities = intensities
        self.design = design
        self.host = host
        self.port = port
        self.logbase = logbase
        self.dash_thread = None
        self.text_output = text_output
        self.sep = sep


    def run(self):
        kwargs = dict(input=self.intensities, design_matrix=self.design, debug=False, port=self.port, host=self.host, logbase=self.logbase, sep=self.sep)
        self.dash_thread = multiprocessing.Process(target=gui_wrapper, kwargs=kwargs)
        self.dash_thread.start()
        linkstr = f"http://{self.host}:{self.port}/"
        QDesktopServices.openUrl(QUrl(linkstr))
        self.text_output.setText(f"Server Running: {linkstr}")
        while self.dash_thread.is_alive():
            time.sleep(2)
        self.text_output.setText(f"")


    def quitserver(self):
        self.dash_thread.terminate()
        self.dash_thread.join()
        self.text_output.setText(f"")






class RDPMSpecIdentifierGUI(QWidget):
    def __init__(self, parent=None):
        super(RDPMSpecIdentifierGUI, self).__init__(parent)
        self.thread = None

        layout = QGridLayout()
        layout.setSpacing(20)
        self.setGeometry(50, 50, 1000, 500)
        self.btn1 = QPushButton("Load Intensity File")
        self.btn1.clicked.connect(self.get_intentity_file)
        layout.addWidget(self.btn1, 1, 0)
        self.file1 = DragAndDropWidget()
        layout.addWidget(self.file1, 1, 1)


        self.btn2 = QPushButton("Load Design File")
        self.btn2.clicked.connect(self.get_design_file)

        layout.addWidget(self.btn2, 2, 0)
        self.file2 = DragAndDropWidget()

        layout.addWidget(self.file2, 2, 1)





        layout.addWidget(QLabel('Host:'), 3, 0)
        self.host = QLineEdit()
        self.host.setText("127.0.0.1")
        self.host.setEnabled(False)
        layout.addWidget(self.host, 3, 1)

        layout.addWidget(QLabel('Port:'), 4, 0)
        self.port = QLineEdit()
        self.port.setText("8080")
        self.port.setEnabled(False)
        layout.addWidget(self.port, 4, 1)

        layout.addWidget(QLabel('log base:'), 5, 0)
        self.logbase = QLineEdit()
        self.logbase.setText("None")
        layout.addWidget(self.logbase, 5, 1)

        layout.addWidget(QLabel('seperator:'), 6, 0)
        self.sep = QLineEdit()
        self.sep.setText(",")
        layout.addWidget(self.sep, 6, 1)


        layout.setAlignment(Qt.AlignTop)
        self.runServerBtn = QPushButton("Run Server")
        self.runServerBtn.clicked.connect(self.run_server)
        layout.addWidget(self.runServerBtn, 7, 0, 1, 2)

        self.killServerBtn = QPushButton("Kill Server")
        self.killServerBtn.clicked.connect(self.kill_server)
        self.killServerBtn.setDisabled(True)

        layout.addWidget(self.killServerBtn, 8, 0, 1, 2)

        svg = HeaderLine()
        layout.addWidget(svg, 0, 0, 1, 2)

        self.running = QLabel("")
        self.running.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.running, 9, 0, 1, 2)

        self.setStyleSheet(STYLE)


        self.setLayout(layout)
        self.setWindowTitle("RDPMSpecIdentifier")
        self.setWindowIcon(QIcon(ICON))

    def closeEvent(self, event):
        if self.thread is not None:
            self.kill_server()
        event.accept()

    def run_server(self):
        port = self.port.text()
        host = self.host.text()
        intenstities = self.file1.text.text()
        design = self.file2.text.text()
        logbase = self.logbase.text()
        logbase = None if logbase == 'None' else int(logbase)
        self.thread = DashRunner(intenstities, design, host, port, logbase, text_output=self.running, sep=self.sep.text())
        self.thread.start()
        self.runServerBtn.setDisabled(True)
        self.killServerBtn.setEnabled(True)


    def kill_server(self):

        if self.thread is not None:

            while self.thread.isRunning():
                self.thread.quitserver()
                self.thread.quit()
                self.thread.wait(2)

        self.killServerBtn.setDisabled(True)
        self.runServerBtn.setEnabled(True)


    def get_intentity_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            'c:\\', "CSV (*.csv *.tsv)")
        self.file1.setText(fname[0])

    def get_design_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            'c:\\', "CSV (*.csv *.tsv)")

        self.file2.setText(fname[0])


def main():
    app = QApplication(sys.argv)
    ex = RDPMSpecIdentifierGUI()
    ex.show()
    sys.exit(app.exec_())


def qt_wrapper(args):
    main()


if __name__ == '__main__':
    main()