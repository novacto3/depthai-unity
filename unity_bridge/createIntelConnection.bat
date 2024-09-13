@ECHO OFF
if [%1]==[] goto :error
python .\intel_hand_tracking_unity_bridge.py -p %1 --internal_frame_width=640 --internal_frame_height=480
goto :end

:error
echo Missing PORT parameter

:end