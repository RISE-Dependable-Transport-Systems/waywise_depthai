# MJPEG, JSON Streaming server + birdseye view

This script allows you to:
- Stream frames via HTTP Server using MJPEG stream
- Stream data via TCP Server

## Installation

```
Ubuntu:
sudo wget -qO- https://docs.luxonis.com/install_dependencies.sh | bash

Raspberry Pi OS:
sudo curl -fL https://docs.luxonis.com/install_dependencies.sh | bash

python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```

To see the streamed frames, open [localhost:8090](http://localhost:8090)

To see the streamed data, use

```
nc localhost 8070
```

On RPi, after running sudo apt upgrade, you might get the error realloc(): invalid pointer\n Aborted when importing cv2 after depthai library. Solution is to downgrade libc6

```
sudo apt install libc6=2.28-10+rpi1
```

