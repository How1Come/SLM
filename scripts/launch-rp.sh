set -eu
rp_project_dir=$(dirname "$0")/..
cd "$rp_project_dir"
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi
exec "${RP_PYTHON:-python}" -m rp.ui \
    --host "${RP_HOST:-127.0.0.1}" \
    --port "${RP_UI_PORT:-8847}" \
    --rp-host 127.0.0.1 \
    --rp-port "${RP_PORT:-8627}" \
    --data-dir "${RP_DATA_DIR:-rp_data}" \
    --model "${RP_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" \
    "$@"
