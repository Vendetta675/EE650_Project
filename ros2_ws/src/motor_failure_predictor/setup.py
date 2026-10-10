from setuptools import find_packages, setup

package_name = 'motor_failure_predictor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lstm_predictor = motor_failure_predictor.lstm_node:main',
            'random_publisher = motor_failure_predictor.random_flight_data_publisher:main',
            'mavlink_bridge = motor_failure_predictor.mavlink_bridge:main',
            'fault_supervisor = motor_failure_predictor.fault_supervisor_node:main'
        ],
    },
)
