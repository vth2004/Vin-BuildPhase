#!/bin/sh
set -eu

for model in "$POSE_MODEL_PATH" "$FACE_LANDMARKER_MODEL_PATH"; do
  if [ ! -f "$model" ]; then
    echo "Required model file is missing: $model" >&2
    exit 1
  fi
done

if [ "${QA_ENSEMBLE_MODE:-K2}" = "K2" ]; then
  for model in "$RTMPOSE_DET_MODEL_PATH" "$RTMPOSE_POSE_MODEL_PATH"; do
    if [ ! -f "$model" ]; then
      echo "Required RTMPose model file is missing: $model" >&2
      exit 1
    fi
  done
fi

exec "$@"
