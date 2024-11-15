from setuptools import find_packages, setup

package_name = 'seg_det'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/detection_launch.py']),
    ],
    py_modules=['seg_det.cam_subscriber',
                'seg_det.unet'],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'det_node = seg_det.cam_subscriber:main',
            'cam_node = seg_det.read_img:main',
        ],
    },
)
