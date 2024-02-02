import sys
from PyQt5.QtCore import Qt, QThread, QUrl, QObject, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QDesktopServices, QTextCursor
import re
from PyQt5.QtWidgets import (
    QFrame,
    QWidget,
    QLabel,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QApplication,
    QPlainTextEdit
)
from io import StringIO
from PyQt5.QtSvg import QSvgWidget
import multiprocessing
import os
import time
from RAPDOR.visualize.appDefinition import app
import socket
import errno
from RAPDOR.visualize.runApp import get_app_layout
from queue import Queue
import logging
import logging.handlers
from multiprocessing import freeze_support   # <---add this

logger = logging.getLogger("RAPDOR")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

FPATH = os.path.abspath(__file__)
BASE = os.path.dirname(FPATH)


STYLESHEET = os.path.join(BASE, "exestyle.css")
with open(STYLESHEET) as handle:
    STYLE = handle.read()

ICON = os.path.join(BASE, "RAPDOR/visualize/assets/favicon.ico")
HEADER = os.path.join(BASE, "RAPDOR/visualize/assets/RAPDOR.svg")
assert os.path.exists(HEADER), f"{HEADER} File does not exist"
assert os.path.exists(ICON), f"{ICON} File does not exist"


class QTextEditLogger(logging.Handler, QObject):
    appendPlainText = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__()
        QObject.__init__(self)
        self.widget = QPlainTextEdit(parent)
        self.widget.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.widget.setReadOnly(True)
        self.appendPlainText.connect(self.widget.appendPlainText)

    def emit(self, record):
        msg = self.format(record)
        self.appendPlainText.emit(msg)




class HeaderLine(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        svg = QSvgWidget(HEADER)
        svg.setMaximumSize(300, 50)
        svg.setMinimumSize(300, 50)
        svg.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        layout.addWidget(svg)
        self.setLayout(layout)






def run_app(debug, port, host, queue):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    app.layout = get_app_layout()
    app.run(debug=debug, port=port, host=host)


class DashRunner(QThread):
    def __init__(self, host, port, text_output, queue):
        super().__init__()
        self.host = host
        self.port = port
        self.dash_thread = None
        self.text_output = text_output
        self.status = None
        self.queue = queue


    def run(self):
        kwargs = dict(debug=False, port=self.port, host=self.host, queue=self.queue)


        self.dash_thread = multiprocessing.Process(target=run_app, kwargs=kwargs)
        self.dash_thread.start()
        linkstr = f"http://{self.host}:{self.port}/"
        QDesktopServices.openUrl(QUrl(linkstr))
        self.text_output.setText(f"Server Running: {linkstr}")
        idx = 1
        while self.dash_thread.is_alive():

            time.sleep(1)
        if self.status == "closed":
            self.status = None
            self.text_output.setText(f"Server terminated")

        else:
            self.text_output.setText(f"Server crashed")


    def quitserver(self):
        self.dash_thread.terminate()
        self.dash_thread.join()
        self.text_output.setText(f"")


class NoAnsiFormatter(logging.Formatter):
    ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

    def format(self, record):
        message = super().format(record)
        return self.strip_ansi_escape(message)

    @staticmethod
    def strip_ansi_escape(s):
        return NoAnsiFormatter.ANSI_ESCAPE.sub('', s)


class LogginThread(QThread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        while True:
            try:
                record = self.queue.get()
                if record is None:  # We send this as a sentinel to tell the listener to quit.
                    break
                logger = logging.getLogger(record.name)
                logger.handle(record)
            except Exception:
                print('Whoops! Problem:', file=sys.stderr)


class ColorButton(QPushButton):
    def __init__(self, *args, color):
        super().__init__(*args)
        self.color = QColor(color)
        assert self.color.isValid()
        self._set_style()

    def _set_style(self):
        style = f"background-color : {self.color.name()}"
        self.setStyleSheet(style)

    def setColor(self, color):
        assert self.color.isValid()
        self.color = color
        self._set_style()


class RAPDORGUI(QWidget):
    def __init__(self, default_port, default_host, parent=None):
        super(RAPDORGUI, self).__init__(parent)
        self.thread = None
        self.queue = multiprocessing.Queue(-1)

        layout = QGridLayout()
        layout.setSpacing(20)
        self.setGeometry(50, 50, 1000, 500)
        idx = 1
        layout.addWidget(QLabel('Host:'), idx, 0)
        self.host = QLineEdit()
        self.host.setText(str(default_host))
        self.host.setEnabled(False)
        layout.addWidget(self.host, idx, 1)
        idx += 1

        layout.addWidget(QLabel('Port:'), idx, 0)
        self.port = QLineEdit()
        self.port.setText(str(default_port))
        self.port.setEnabled(False)
        layout.addWidget(self.port, idx, 1)
        idx += 1

        p = QGridLayout()
        layout.setAlignment(Qt.AlignTop)
        self.runServerBtn = QPushButton("Run Server")
        self.runServerBtn.clicked.connect(self.run_server)
        p.addWidget(self.runServerBtn, 0, 0, 1, 1)

        self.killServerBtn = QPushButton("Kill Server")
        self.killServerBtn.clicked.connect(self.kill_server)
        self.killServerBtn.setDisabled(True)
        p.addWidget(self.killServerBtn, 0, 1, 1, 1)
        layout.addLayout(p, idx, 0, 1, 2)

        idx += 1

        svg = HeaderLine()
        layout.addWidget(svg, 0, 0, 1, 2)

        self.running = QLabel("")
        self.running.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.running, idx, 0, 1, 2)
        idx += 1
        self.openBrowserBtn = QPushButton("Open Browser")
        self.openBrowserBtn.setDisabled(True)

        self.openBrowserBtn.clicked.connect(self.open_browser)
        layout.addWidget(self.openBrowserBtn, idx, 0, 1, 2)
        idx += 1

        # self.textedit = QTextEdit(self)
        # layout.addWidget(self.textedit, idx, 0, 1, 2)

        self.setStyleSheet(STYLE)

        logTextBox = QTextEditLogger(self)
        self.loggingthread = LogginThread(self.queue)
        self.loggingthread.start()

        # You can format what is printed to text box
        formatter = NoAnsiFormatter('%(levelname)s - %(message)s')
        logTextBox.setFormatter(
            formatter)
        logging.getLogger().addHandler(logTextBox)
        # You can control the logging level
        logging.getLogger().setLevel(logging.DEBUG)
        layout.addWidget(logTextBox.widget, idx, 0, 1, 2)


        self.setLayout(layout)
        self.setWindowTitle("RAPDOR")
        self.setWindowIcon(QIcon(ICON))

    def open_browser(self):
        port = self.port.text()
        host = self.host.text()
        linkstr = f"http://{host}:{port}/"
        QDesktopServices.openUrl(QUrl(linkstr))


    def select_latest(self, idx):
        def fct():
            box = self.sep_boxes[idx]
            any_checked = False
            for idx2 in range(self.buttons.count()):
                btn = self.sep_boxes[idx2]
                if btn.isChecked() and btn.text() != box.text():
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
                    any_checked = True
            if not any_checked:
                box.setChecked(True)

        return fct

    def closeEvent(self, event):
        if self.thread is not None:
            self.kill_server()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.loggingthread.quit()
        self.queue.put_nowait(None)

        event.accept()

    def run_server(self):
        port = self.port.text()
        host = self.host.text()
        self.thread = DashRunner(host, port, text_output=self.running, queue=self.queue)
        self.thread.start()
        self.runServerBtn.setDisabled(True)
        self.killServerBtn.setEnabled(True)
        self.openBrowserBtn.setEnabled(True)


    def kill_server(self):
        if self.thread is not None:

            while self.thread.isRunning():
                self.thread.dash_thread.terminate()
                self.thread.dash_thread.join()
                self.thread.status = "closed"
                self.thread.quitserver()
                self.thread.quit()
                self.thread.wait(2)
        self.killServerBtn.setDisabled(True)
        self.runServerBtn.setEnabled(True)
        self.openBrowserBtn.setDisabled(True)



    @pyqtSlot(str)
    def append_text(self, text):
        self.textedit.moveCursor(QTextCursor.End)
        self.textedit.insertPlainText( text )


class WriteStream(object):
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        pass


class MyReceiver(QObject):
    mysignal = pyqtSignal(str)

    def __init__(self, queue, *args, **kwargs):
        QObject.__init__(self, *args, **kwargs)
        self.queue = queue

    @pyqtSlot()
    def run(self):
        while True:
            text = self.queue.get()
            self.mysignal.emit(text)




def main():
    port = 49372
    host = "127.0.0.1"
    freeze_support()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    use = True
    while use:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((host, port))
            use = False
        except socket.error as e:
            s.close()
            if e.errno == errno.EADDRINUSE:
                port +=1
            else:
                # something else raised the socket.error exception
                print(e)
        finally: s.close()

    app = QApplication(sys.argv)
    ex = RAPDORGUI(default_port=port, default_host=host)
    ex.show()
    sys.exit(app.exec_())



if __name__ == '__main__':
    main()