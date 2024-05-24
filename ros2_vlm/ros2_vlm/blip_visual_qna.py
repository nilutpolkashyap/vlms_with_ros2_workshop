#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from PIL import Image
from .modules.blipvisual import BlipVQA
import openvino as ov
from pathlib import Path
import sys
import os

current_dir = os.getcwd()
additional_path = '/src/ros2_vlm/ros2_vlm/modules'
updated_path = os.path.join(current_dir + additional_path)

class QuestionImageProcessor(Node):
    def __init__(self):
        super().__init__('question_image_processor_node')

        self.core = ov.Core()

        self.declare_parameter('device', 'GPU.0')
        self.declare_parameter('question', 'What is in the image?')
        self.declare_parameter('image_path', '')

        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.question = self.get_parameter('question').get_parameter_value().string_value
        self.image_path = self.get_parameter('image_path').get_parameter_value().string_value

        if not self.image_path:
            self.get_logger().error('image_path parameter is required.')
            return

        self.get_logger().info(f'Inference Device: {self.device}')
        self.get_logger().info(f'Question: {self.question}')
        self.get_logger().info(f'Image Path: {self.image_path}')

        self.irs_path = Path("openvino_irs")

        self.vision_model_path = updated_path / self.irs_path / f"blip_vision_model.xml"
        self.text_encoder_path = updated_path / self.irs_path / f"blip_text_encoder.xml"
        self.text_decoder_path = updated_path / self.irs_path / f"blip_text_decoder_with_past.xml"

        self.blip_inference = BlipVQA(self.core, self.vision_model_path, self.text_encoder_path, self.text_decoder_path, self.device)

        try:
            answer, inference_time = self.blip_inference.generate_answer(self.image_path, self.question, max_length=20)

            print(f"Generated Answer: {answer}")
            print(f"Inference Time: {inference_time} seconds")
        except Exception as e:
            self.get_logger().error(f'Failed to load image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = QuestionImageProcessor()

    if rclpy.ok():
        rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
