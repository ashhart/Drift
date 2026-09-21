#!/bin/sh
# Runs on every NON-head tensor-parallel host while taps are active. The owner's handoff connector writes
# one blob per rank into that host's /dev/shm; Drift reads rank 0 only (MLA latents are replicated),
# so the other ranks' drift-* exports are deleted as soon as they are complete. Nothing else is touched.
# Stops after $1 seconds (default 7200) or when /tmp/drift-janitor.stop exists.
exec 9> /tmp/drift-janitor.lock; flock -n 9 || exit 0          # one janitor per host; later starts are no-ops
end=$(( $(date +%s) + ${1:-7200} ))
rm -f /tmp/drift-janitor.stop
while [ "$(date +%s)" -lt "$end" ] && [ ! -e /tmp/drift-janitor.stop ]; do
  # only COMPLETE exports (rank*.ready present) are removed: a long document's blob takes longer than any fixed age to write
  for d in /dev/shm/glm53-handoff/drift-*; do
    [ -d "$d" ] && ls "$d"/rank*.ready > /dev/null 2>&1 && rm -rf "$d"
  done
  sleep 3
done
