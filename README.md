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

Make sure to add udev rules:
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

Finally:
./install.sh
```

## Usage

Run the application

```
./camerastart.sh
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

