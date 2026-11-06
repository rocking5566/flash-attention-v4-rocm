#!/usr/bin/env bash
# Run from the local checkout. Always return logs/artifacts before a later sync.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
# Serialize sync/run/retrieval, including callers started in separate exec cells.
exec 9>build/remote.lock
flock 9
source .remote.env
[[ $REMOTE_HOST != CHANGE_ME && $REMOTE_CONTAINER != CHANGE_ME ]] || {
    echo 'Configure .remote.env before syncing.' >&2; exit 2;
}
[[ ${REMOTE_PATH##*/} == FAv4 && ${CONTAINER_PATH##*/} == FAv4 ]] || {
    echo 'Refusing --delete sync outside the dedicated FAv4 directory' >&2; exit 2;
}
[[ $# -gt 0 ]] || { echo 'usage: scripts/remote.sh COMMAND [ARGS...]' >&2; exit 2; }
rsync -az --delete $RSYNC_EXCLUDES --exclude .remote.env ./ "$REMOTE_HOST:$REMOTE_PATH/"
printf -v task_cmd '%q ' "$@"
printf -v inner 'cd %q && %s' "$CONTAINER_PATH" "$task_cmd"
printf -v quoted_inner '%q' "$inner"
status=0
ssh -o LogLevel=ERROR "$REMOTE_HOST" \
    "docker start $REMOTE_CONTAINER >/dev/null && docker exec -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES $REMOTE_CONTAINER bash -lc $quoted_inner" || status=$?
# results/ holds every generated artifact that should survive a future --delete.
# Some GPU hosts do not expose an SFTP subsystem; try legacy SCP first.
fetch=${FAV4_FETCH:-results}
[[ $fetch != none ]] || exit "$status"
[[ $fetch == results || $fetch == results/* ]] || exit 2
mkdir -p "$(dirname "$fetch")"
scp -O -q -r "$REMOTE_HOST:$REMOTE_PATH/$fetch" "$(dirname "$fetch")/" ||
    scp -q -r "$REMOTE_HOST:$REMOTE_PATH/$fetch" "$(dirname "$fetch")/" || status=1
exit "$status"
