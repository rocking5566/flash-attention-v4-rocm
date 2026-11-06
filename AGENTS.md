# Remote GPU dev

Read, edit and run git here; build, test and profile on the remote server.

Read `.remote.env` for the target (never hardcode — hosts move). No
`.remote.env`? Copy `.remote.env.example` and ask which host; never sync to
its placeholder.

Sync, then run — this also starts `$REMOTE_CONTAINER` if it's stopped:

    rsync -az --delete $RSYNC_EXCLUDES ./ "$REMOTE_HOST:$REMOTE_PATH/"
    ssh "$REMOTE_HOST" "docker start $REMOTE_CONTAINER && docker exec -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES $REMOTE_CONTAINER \
        bash -lc 'cd $CONTAINER_PATH && <cmd>'"

Wrote a file on the remote? scp it back in that same step — the next sync's
`--delete` wipes it. Never commit from the remote; its `.git` is excluded and
stale.