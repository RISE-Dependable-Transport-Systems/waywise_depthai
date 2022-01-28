import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from socketserver import ThreadingMixIn
from time import sleep
import depthai as dai
import cv2
from PIL import Image
import math
import numpy as np
import blobconverter

HTTP_SERVER_PORT = 8090

class TCPServerRequest(socketserver.BaseRequestHandler):
    def handle(self):
        ## Handle is called each time a client is connected
        ## When OpenDataCam connects, do not return - instead keep the connection open and keep streaming data
        ## First send HTTP header
        #header = 'HTTP/1.0 200 OK\r\nServer: Mozarella/2.2\r\nAccept-Range: bytes\r\nConnection: close\r\nMax-Age: 0\r\nExpires: 0\r\nCache-Control: no-cache, private\r\nPragma: no-cache\r\nContent-Type: application/json\r\n\r\n'
        #self.request.send(header.encode())
        while True:
            sleep(0.1)
            if hasattr(self.server, 'datatosend'):
                self.request.send(self.server.datatosend.encode() + "\r\n".encode())


# HTTPServer MJPEG
class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        while True:
            sleep(0.1)
            if hasattr(self.server, 'frametosend'):
                image = Image.fromarray(cv2.cvtColor(self.server.frametosend, cv2.COLOR_BGR2RGB))
                stream_file = BytesIO()
                image.save(stream_file, 'JPEG')
                self.wfile.write("--jpgboundary".encode())

                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(stream_file.getbuffer().nbytes))
                self.end_headers()
                image.save(self.wfile, 'JPEG')


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass
    
# start TCP data server
server_TCP = socketserver.TCPServer(('', 8070), TCPServerRequest)
th = threading.Thread(target=server_TCP.serve_forever)
th.daemon = True
th.start()


# start MJPEG HTTP Server
server_HTTP = ThreadedHTTPServer(('', 8090), VideoStreamHandler)
th2 = threading.Thread(target=server_HTTP.serve_forever)
th2.daemon = True
th2.start()

class BirdFrame():
    max_z = 6
    min_z = 0
    max_x = 1.0
    min_x = -1.0

    def __init__(self):
        self.distance_bird_frame = self.make_bird_frame()

    def make_bird_frame(self):
        fov = 68.7938
        min_distance = 0.827
        frame = np.zeros((320, 100, 3), np.uint8)
        min_y = int((1 - (min_distance - self.min_z) / (self.max_z - self.min_z)) * frame.shape[0])
        cv2.rectangle(frame, (0, min_y), (frame.shape[1], frame.shape[0]), (70, 70, 70), -1)

        alpha = (180 - fov) / 2
        center = int(frame.shape[1] / 2)
        max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
        fov_cnt = np.array([
            (0, frame.shape[0]),
            (frame.shape[1], frame.shape[0]),
            (frame.shape[1], max_p),
            (center, frame.shape[0]),
            (0, max_p),
            (0, frame.shape[0]),
        ])
        cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
        return frame

    def calc_x(self, val):
        norm = min(self.max_x, max(val, self.min_x))
        center = (norm - self.min_x) / (self.max_x - self.min_x) * self.distance_bird_frame.shape[1]
        bottom_x = max(center - 2, 0)
        top_x = min(center + 2, self.distance_bird_frame.shape[1])
        return int(bottom_x), int(top_x)

    def calc_z(self, val):
        norm = min(self.max_z, max(val, self.min_z))
        center = (1 - (norm - self.min_z) / (self.max_z - self.min_z)) * self.distance_bird_frame.shape[0]
        bottom_z = max(center - 2, 0)
        top_z = min(center + 2, self.distance_bird_frame.shape[0])
        return int(bottom_z), int(top_z)

    def parse_frame(self, frame, detections):
        bird_frame = self.distance_bird_frame.copy()
        for detection in detections:
            left, right = self.calc_x(detection.spatialCoordinates.x / 1000)
            top, bottom = self.calc_z(detection.spatialCoordinates.z / 1000)
            cv2.rectangle(bird_frame, (left, top), (right, bottom), (0, 255, 0), 2)

        return np.hstack((frame, bird_frame))

def create_pipeline(model_name):
    pipeline = dai.Pipeline()
    #pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4_2)

    # Define sources and outputs
    camRgb = pipeline.createColorCamera()
    spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()

    xoutRgb = pipeline.createXLinkOut()
    camRgb.preview.link(xoutRgb.input)
    xoutNN = pipeline.createXLinkOut()

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")

    # Properties
    camRgb.setPreviewSize(544, 320)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # Setting node configs
    stereo.initialConfig.setConfidenceThreshold(255)

    spatialDetectionNetwork.setBlobPath(blobconverter.from_zoo(name=model_name, shaves=6))
    #spatialDetectionNetwork.setBlobPath(blobconverter.from_zoo("person-detection-retail-0013", shaves=6))
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    camRgb.preview.link(spatialDetectionNetwork.input)

    spatialDetectionNetwork.out.link(xoutNN.input)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)

    return pipeline
    
# Pipeline is defined, now we can connect to the device
with dai.Device() as device:
    cams = device.getConnectedCameras()
    depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams

    # Start pipeline. Can select other NN here
    device.startPipeline(create_pipeline("person-detection-retail-0013"))

    print(f"DepthAI is up & running. Navigate to 'localhost:{str(HTTP_SERVER_PORT)}' with Chrome to see the mjpeg stream")

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    detections = []
    bf = BirdFrame()

    while True:
        frame = previewQueue.get().getCvFrame()
        inDet = detectionNNQueue.tryGet()
        
        if inDet is not None:
    	    detections = inDet.detections

        img_h = frame.shape[0]
        img_w = frame.shape[1]
        for detection in detections:
            print(detection)
            left, top = int(detection.xmin * img_w), int(detection.ymin * img_h)
            right, bottom = int(detection.xmax * img_w), int(detection.ymax * img_h)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        annotated_frame = bf.parse_frame(frame, detections)
        #server_TCP.datatosend = json.dumps([detection.get_dict() for detection in detections])
        server_TCP.datatosend = json.dumps([detection.getData() for detection in detections])
        server_HTTP.frametosend = annotated_frame
        cv2.imshow('previewout', annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break

del pipeline
del device
