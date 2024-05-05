@ECHO OFF
if [%1]==[] goto :error
python .\depthai_hand_tracking_unity_bridge.py -p %1 -xyz --use_world_landmarks --internal_frame_height=1024 --lm_model=full  -e -i rgb_laconic

:error
echo Missing PORT parameter