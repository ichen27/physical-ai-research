#!/bin/zsh
# Takes pictures with the camera on the arm, and no other camera, when asked.
#
# macOS only lets a program use a camera if the person at the computer started
# it. Leave this running in Terminal on the Mac the arm's camera is plugged
# into, and other programs (including ones started over a remote login) can
# ask it for a picture: each time a file named "take" appears in this script's
# folder, it takes one picture and saves it as shots/<name>.jpg, where <name>
# is the text inside that file, then writes shots/<name>.done.
# understudy/camera_feed.py (WatcherCamera) is the other side of this.
#
# Set up once:
#   mkdir -p ~/understudy-camera && cp scripts/camera_watch.sh ~/understudy-camera/camera-watch.sh
#   put an ffmpeg binary in ~/understudy-camera/ (or have ffmpeg on the PATH)
# Run:   ~/understudy-camera/camera-watch.sh        Stop: Control-C.
#
# CAMERA is the start of the camera's name as macOS lists it
# (ffmpeg -f avfoundation -list_devices true -i ""). A laptop's own camera or
# a phone's will not match it, and nothing else is ever opened.
CAMERA=${UNDERSTUDY_CAMERA:-"Innomaker-U20CAM-720P"}

cd "$(dirname "$0")"
if [[ -x ./ffmpeg ]]; then FFMPEG=./ffmpeg; else FFMPEG=ffmpeg; fi
device=$($FFMPEG -hide_banner -f avfoundation -list_devices true -i "" 2>&1 \
  | sed -n '/video devices/,/audio devices/p' | grep -F "] $CAMERA" | sed 's/^.*\] //')
if [[ -z "$device" || $(print -r -- "$device" | wc -l) -ne 1 ]]; then
  echo "Expected exactly one camera whose name starts with \"$CAMERA\". Is the arm's camera plugged in?"
  exit 1
fi
mkdir -p shots
rm -f take
echo "Camera watcher running, using \"${device}\" only. It takes a picture only when asked. Stop it with Control-C."
while true; do
  if [[ -e take ]]; then
    name=$(head -c 60 take | tr -cd "A-Za-z0-9_-")
    rm -f take
    [[ -n "$name" ]] || name="shot"
    # The first frames after a camera starts are badly exposed, so a dozen are skipped.
    $FFMPEG -hide_banner -loglevel error -f avfoundation -pixel_format uyvy422 -framerate 30 \
      -i "${device}:none" -vf "select=gte(n\,12)" -frames:v 1 -q:v 2 -y "shots/$name.jpg" &
    pid=$!
    for i in {1..30}; do kill -0 $pid 2>/dev/null || break; sleep 0.5; done
    kill -9 $pid 2>/dev/null
    if [[ -s "shots/$name.jpg" ]]; then echo "$(date +%H:%M:%S) took $name"; else echo "$(date +%H:%M:%S) FAILED $name"; fi
    echo done > "shots/$name.done"
  fi
  sleep 0.2
done
