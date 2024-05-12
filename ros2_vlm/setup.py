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
    description='ROS 2 OpenCV Python Package',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'grounded_sam = ros2_vlm.grounded_sam:main',
        ],
    },
)
