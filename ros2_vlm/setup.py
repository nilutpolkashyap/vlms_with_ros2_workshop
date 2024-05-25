from setuptools import setup
import os
from glob import glob 

package_name = 'ros2_vlm'
modules = "ros2_vlm/modules"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, modules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nilutpol Kashyap',
    maintainer_email='nilutpolkashyap@todo.todo',
    description='Application of Vision Language Models with ROS 2 workshop',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'grounded_sam = ros2_vlm.grounded_sam:main',
            'blip_visual_qna = ros2_vlm.blip_visual_qna:main',
        ],
    },
)
