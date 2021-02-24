# MJPEG, JSON Streaming server + birdseye view

This script allows you to:
- Stream frames via HTTP Server using MJPEG stream
- Stream data via TCP Server

## Installation

```
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
