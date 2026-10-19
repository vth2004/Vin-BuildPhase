#!/usr/bin/env bash
set -Eeuo pipefail

# Deploy Vin-BuildPhase source over SSH, then let Coolify build and run it.
# Run in Git Bash/WSL: ./deploy-vps.sh
# On Windows: powershell -ExecutionPolicy Bypass -File .\deploy-vps.ps1
#
# Layout on the VPS is the same shape as the p036-coolify service:
#   /opt/vin-buildphase/            -> source (backend/, frontend/, compose.yaml)
#   /opt/vin-buildphase/deploy-data -> SQLite + dataset files (persisted)
#   /opt/vin-buildphase/deploy-models -> 4 model files (persisted, read-only)
#
# Configuration lives in .deploy.env next to this script (see .deploy.env.example).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ROOT_DIR/.deploy.env" ]] && source "$ROOT_DIR/.deploy.env"

VPS_HOST="${VPS_HOST:-103.72.56.152}"
VPS_USER="${VPS_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/opt/vin-buildphase}"
TARGET="${VPS_USER}@${VPS_HOST}"
DEPLOY_TIMEOUT_SECONDS="${DEPLOY_TIMEOUT_SECONDS:-1800}"
DEPLOY_POLL_SECONDS="${DEPLOY_POLL_SECONDS:-10}"
COOLIFY_URL="${COOLIFY_URL:-http://${VPS_HOST}:8000}"
COOLIFY_SERVICE_UUID="${COOLIFY_SERVICE_UUID:-}"
PUBLIC_FRONTEND_URL="${PUBLIC_FRONTEND_URL:-http://landmarkqa.duckdns.org}"
PUBLIC_BACKEND_URL="${PUBLIC_BACKEND_URL:-http://landmarkqabe.duckdns.org}"
# auto = upload models only when they are missing on the VPS; true = always; false = never
SYNC_MODELS="${SYNC_MODELS:-auto}"

ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/vin-buildphase.XXXXXX.tar.gz")"
MODEL_ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/vin-buildphase-models.XXXXXX.tar.gz")"
SSH_OPTS=(-i "$SSH_KEY" -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)

log() {
    printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

cleanup() { rm -f "$ARCHIVE" "$MODEL_ARCHIVE"; }
on_error() {
    local status=$?
    printf '\n[ERROR] Deployment stopped at line %s (exit %s).\n' "$1" "$status" >&2
    exit "$status"
}

trap cleanup EXIT
trap 'on_error $LINENO' ERR

for command in ssh scp tar curl base64 node; do
    command -v "$command" >/dev/null || { echo "Missing: $command" >&2; exit 1; }
done
[[ -f "$SSH_KEY" ]] || { echo "SSH key not found: $SSH_KEY" >&2; exit 1; }
[[ -n "$COOLIFY_SERVICE_UUID" ]] || {
    echo "COOLIFY_SERVICE_UUID is missing. Create the Coolify service first and put its UUID in .deploy.env." >&2
    exit 1
}
[[ -n "${COOLIFY_TOKEN:-}" ]] || { echo "COOLIFY_TOKEN missing in .deploy.env" >&2; exit 1; }

log "1/6 Checking SSH access to ${TARGET}"
ssh "${SSH_OPTS[@]}" "$TARGET" 'exit 0' || {
    echo "SSH access failed; nothing was uploaded." >&2
    exit 1
}
log "SSH access confirmed"

log "2/6 Packing source files"
tar --exclude=node_modules --exclude=.venv --exclude=__pycache__ --exclude='*.pyc' \
    --exclude='*.pt' --exclude='*.task' --exclude='*.onnx' --exclude='*.zip' \
    --exclude=backend/data --exclude=frontend/dist --exclude=deploy-data --exclude=deploy-models \
    -C "$ROOT_DIR" -czf "$ARCHIVE" backend frontend compose.yaml
log "Archive ready: $(du -h "$ARCHIVE" | awk '{print $1}')"

log "3/6 Uploading source to ${REMOTE_DIR}"
scp "${SSH_OPTS[@]}" "$ARCHIVE" "${TARGET}:/tmp/vin-buildphase-deploy.tar.gz"

log "4/6 Preparing source on VPS (keeping deploy-data, deploy-models and .env)"
ssh "${SSH_OPTS[@]}" "$TARGET" "REMOTE_DIR='$REMOTE_DIR' bash -s" <<'REMOTE'
set -Eeuo pipefail
mkdir -p "$REMOTE_DIR/deploy-data" "$REMOTE_DIR/deploy-models"
# Replace only the source tree; user data and models must survive a redeploy.
find "$REMOTE_DIR" -mindepth 1 -maxdepth 1 \
    ! -name deploy-data ! -name deploy-models ! -name .env -exec rm -rf {} +
tar -xzf /tmp/vin-buildphase-deploy.tar.gz -C "$REMOTE_DIR"
rm -f /tmp/vin-buildphase-deploy.tar.gz
REMOTE

log "5/6 Syncing model files into ${REMOTE_DIR}/deploy-models"
models_needed="false"
if [[ "$SYNC_MODELS" == "true" ]]; then
    models_needed="true"
elif [[ "$SYNC_MODELS" == "auto" ]]; then
    remote_models_ok="$(ssh "${SSH_OPTS[@]}" "$TARGET" "REMOTE_DIR='$REMOTE_DIR' bash -s" <<'REMOTE'
set -e
ok=1
for f in yolo26s-pose.pt face_landmarker.task \
         yolox_m_8xb8-300e_humanart-c2c7a14a.onnx \
         rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx; do
  [ -s "$REMOTE_DIR/deploy-models/$f" ] || ok=0
done
printf '%s' "$ok"
REMOTE
)"
    if [[ "$remote_models_ok" != "1" ]]; then
        models_needed="true"
        log "Some model files are missing on the VPS"
    else
        log "All model files already present on the VPS (SYNC_MODELS=$SYNC_MODELS)"
    fi
fi
if [[ "$models_needed" == "true" ]]; then
    [[ -d "$ROOT_DIR/deploy-models" ]] || {
        echo "deploy-models/ not found locally. Stage the four model files before deploying." >&2
        exit 1
    }
    tar -C "$ROOT_DIR/deploy-models" -czf "$MODEL_ARCHIVE" .
    log "Uploading model files ($(du -h "$MODEL_ARCHIVE" | awk '{print $1}'))"
    scp "${SSH_OPTS[@]}" "$MODEL_ARCHIVE" "${TARGET}:/tmp/vin-buildphase-models.tar.gz"
    ssh "${SSH_OPTS[@]}" "$TARGET" "REMOTE_DIR='$REMOTE_DIR' bash -s" <<'REMOTE'
set -Eeuo pipefail
tar -xzf /tmp/vin-buildphase-models.tar.gz -C "$REMOTE_DIR/deploy-models"
rm -f /tmp/vin-buildphase-models.tar.gz
REMOTE
fi

log "6/6 Updating Coolify service ${COOLIFY_SERVICE_UUID}"
compose="$(cat <<EOF
services:
  backend:
    build:
      context: ${REMOTE_DIR}/backend
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      LANDMARK_DATA_DIR: /app/data
      POSE_MODEL_PATH: /models/yolo26s-pose.pt
      FACE_LANDMARKER_MODEL_PATH: /models/face_landmarker.task
      RTMPOSE_DET_MODEL_PATH: /models/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx
      RTMPOSE_POSE_MODEL_PATH: /models/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx
      QA_ENSEMBLE_MODE: K2
      CORS_ORIGINS: ${PUBLIC_FRONTEND_URL}
      MAX_VIDEO_FRAMES: 10000
      MAX_DECODED_VIDEO_BYTES: 5368709120
      SERVICE_URL_BACKEND_8000: /
    volumes:
      - ${REMOTE_DIR}/deploy-data:/app/data
      - ${REMOTE_DIR}/deploy-models:/models:ro
    healthcheck:
      test:
        - CMD
        - python
        - -c
        - "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5)"
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 90s
  frontend:
    build:
      context: ${REMOTE_DIR}/frontend
      dockerfile: Dockerfile
      args:
        VITE_API_URL: /api
    restart: unless-stopped
    environment:
      SERVICE_URL_FRONTEND_80: /
    depends_on:
      backend:
        condition: service_healthy
EOF
)"

printf '%s\n' "$compose" \
    | ssh "${SSH_OPTS[@]}" "$TARGET" "cat > '$REMOTE_DIR/compose.coolify.yaml'"

compose_base64="$(printf '%s' "$compose" | base64 | tr -d '\r\n')"

curl --fail-with-body --silent --show-error -X PATCH \
    -H "Authorization: Bearer $COOLIFY_TOKEN" -H 'Content-Type: application/json' \
    --data "{\"docker_compose_raw\":\"$compose_base64\",\"instant_deploy\":false,\"urls\":[{\"name\":\"frontend\",\"url\":\"$PUBLIC_FRONTEND_URL:80\"},{\"name\":\"backend\",\"url\":\"$PUBLIC_BACKEND_URL:8000\"}]}" \
    "$COOLIFY_URL/api/v1/services/$COOLIFY_SERVICE_UUID" >/dev/null

log "Compose saved. Sending deploy request to Coolify"
curl --fail-with-body --silent --show-error -X POST \
    -H "Authorization: Bearer $COOLIFY_TOKEN" \
    "$COOLIFY_URL/api/v1/deploy?uuid=$COOLIFY_SERVICE_UUID&force=true"

deadline=$((SECONDS + DEPLOY_TIMEOUT_SECONDS))

http_status() {
    curl --connect-timeout 3 --max-time 5 --silent --output /dev/null \
        --write-out '%{http_code}' "$1" || true
}

log "Waiting for Coolify to finish (up to ${DEPLOY_TIMEOUT_SECONDS}s)"
while (( SECONDS < deadline )); do
    service_json="$(curl --fail-with-body --silent --show-error \
        -H "Authorization: Bearer $COOLIFY_TOKEN" \
        "$COOLIFY_URL/api/v1/services/$COOLIFY_SERVICE_UUID")"
    frontend_url="$(printf '%s' "$service_json" | node -e 'let b="";process.stdin.on("data",d=>b+=d).on("end",()=>{const s=JSON.parse(b);const a=(s.applications||[]).find(x=>x.name==="frontend");process.stdout.write(a?.fqdn||"")})')"
    backend_url="$(printf '%s' "$service_json" | node -e 'let b="";process.stdin.on("data",d=>b+=d).on("end",()=>{const s=JSON.parse(b);const a=(s.applications||[]).find(x=>x.name==="backend");process.stdout.write(a?.fqdn||"")})')"
    # Coolify's FQDN includes the target container port; visitors use the proxy
    # on normal HTTP/HTTPS ports, not the container's internal port.
    frontend_url="${frontend_url/:80/}"
    backend_url="${backend_url/:8000/}"
    frontend_status="000"
    backend_status="000"
    [[ -n "$frontend_url" ]] && frontend_status="$(http_status "$frontend_url")"
    [[ -n "$backend_url" ]] && backend_status="$(http_status "${backend_url%/}/api/health")"
    if [[ "$frontend_status" == "200" && "$backend_status" == "200" ]]; then
        log "Web is ready"
        printf '\nFrontend: %s\nBackend health: %s/api/health\n' "$frontend_url" "${backend_url%/}"
        exit 0
    fi
    printf '[%s] Waiting: frontend HTTP %s, backend health HTTP %s\n' \
        "$(date '+%H:%M:%S')" "${frontend_status:-000}" "${backend_status:-000}"
    sleep "$DEPLOY_POLL_SECONDS"
done

printf '\n[ERROR] Coolify did not make the web reachable within %ss. Check vin-buildphase-coolify logs in Coolify.\n' \
    "$DEPLOY_TIMEOUT_SECONDS" >&2
exit 1
