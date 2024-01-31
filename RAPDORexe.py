from RAPDOR.visualize.appDefinition import app
from RAPDOR.visualize.runApp import get_app_layout
import threading
import webbrowser
import time
import socket
import errno


def open_browser(address, delay):
    time.sleep(delay)
    webbrowser.open(address)

# Start a separate thread for opening the browser

app.layout = get_app_layout()


if __name__ == '__main__':
    port = 49372
    host = "127.0.0.1"


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
    s.close()
    address = f"http://{host}:{port}"
    thread = threading.Thread(target=open_browser, args=(address, 5))
    thread.start()
    app.run(debug=False, port=port, host=host, threaded=True)
    thread.join()
