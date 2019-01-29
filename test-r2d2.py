#!/usr/bin/python3
#
# A bit more complete example of camera array rendering
# (rendering only, no TensorRT inferencing)
#
# git clone https://github.com/dchichkov/bullet3
# cd bullet3
# ./build_cmake_pybullet_double.sh
# cd ..
# export PYTHONPATH=bullet3/build_cmake/examples/pybullet

import pybullet as p, numpy as np

p.connect(p.DIRECT, options='--width=160 --height=160 --cameraArraySize=16')

cameraArraySize,width,height = 16,160, 160
viewMatrices, projectionMatrices = [], []
for yaw in range(0,10 * cameraArraySize,10):
        viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = [0,0,0], distance = 1, yaw = yaw, pitch = -10, roll = 0, upAxisIndex = 2)
        projectionMatrix = p.computeProjectionMatrixFOV(fov = 60, aspect = width / height, nearVal = 0.01, farVal = 100)
        viewMatrices.append(viewMatrix)
        projectionMatrices.append(projectionMatrix)

viewMatrices = np.array(viewMatrices, dtype=np.float32)
projectionMatrices = np.array(projectionMatrices, dtype=np.float32)

plane = p.loadURDF("plane.urdf")
p.loadURDF("plane.urdf",[0,0,-1])
p.loadURDF("r2d2.urdf")
p.loadURDF("duck_vhacd.urdf")

cameraArraySize,width,height,rgb,featureLength,features = p.getCameraArrayImage(cameraArraySize=cameraArraySize,width=width,height=height,featureLength=2000,viewMatrices=viewMatrices,projectionMatrices=projectionMatrices)

from PIL import Image
print(rgb.shape)
Image.frombuffer('RGB', (width,height*cameraArraySize), rgb, 'raw').save("out.png")

print(featureLength,features)
