from setuptools import setup

package_name = 'bag_to_images'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Frank-JJ',
    maintainer_email='fjens20@student.sdu.dk',
    description='Extracts images from ROS 2 Rosbag using topic "image_raw"',
    license='AGPL-3.0 license',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bag_to_images = bag_to_images.bag_to_images:main'
        ],
    },
)
