import rclpy
from rclpy.node import Node
import cv2
import torch
import openvino as ov
from .modules.groundedsam import ImageProcessor
from PIL import Image

class VideoDisplayNode(Node):
    def __init__(self):
        super().__init__('video_display_node')
        
        self.declare_parameter('video_source', '/dev/video0')
        self.declare_parameter('isSegment', False)
        self.declare_parameter('detectionList', ['person'])

        self.video_source = self.get_parameter('video_source').value
        self.isSegment = self.get_parameter('isSegment').value
        self.detectionList = self.get_parameter('detectionList').value

        self.get_logger().info(f'Video source: {self.video_source}')
        self.get_logger().info(f'isSegment: {self.isSegment}')
        self.get_logger().info('My list:')
        for item in self.detectionList:
            self.get_logger().info(f' - {item}')

        self.cap = cv2.VideoCapture(self.video_source)

        self.core = ov.Core()
        self.device = "CPU" #"GPU.0"
        self.processor = ImageProcessor(self.core, self.device)

    def __del__(self):
        self.cap.release()

    def run(self):
        ret, frame = self.cap.read()
        if ret:
            self.pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.processed_image = self.processor.process_image(self.pil_image, self.detectionList, self.isSegment)
            cv2.imshow('Video Frame', self.processed_image)
            cv2.waitKey(0)

def main(args=None):
    rclpy.init(args=args)
    node = VideoDisplayNode()
    node.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
