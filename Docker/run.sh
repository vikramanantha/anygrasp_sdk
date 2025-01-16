docker run -it  \
    --gpus all \
    -e DISPLAY=:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/../:/workspace/grasp \
    -w /workspace/grasp/grasp_detection \
    --mac-address=02:42:ac:11:00:02 \
    grasp_sdk:latest bash


