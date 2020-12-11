#! /bin/bash

gnome-terminal  --window -e 'bash -c "roscore;exec bash"' \
--tab -e 'bash -c "sleep 2s;rosrun tello_control tello_state.py;exec bash"' \
--tab -e 'bash -c "sleep 2s;rosrun tello_control heat.py;exec bash"' \
--tab -e 'bash -c "sleep 2s;rosrun tello_control tello_picture.py;exec bash"' \
--tab -e 'bash -c "sleep 2s;rosrun tello_control tello_yolo.py;exec bash"' 

