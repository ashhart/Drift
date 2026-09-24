# Held-out reverse recall

A result counts as held out only if the agent that builds and tunes the system never saw the test set. The agent cannot generate that set, so the owner does it with three files and one command, and shares only the scorer's aggregate output.

The rule is the one preregistered in [the reverse recall confirmation](REVERSE_RECALL_CONFIRMATION.md). P1 is a one-sided sign test at p < 0.05, linked against no_link, and P2 is at least 18 of 24 linked. The configuration is the same too: whole context at 3 copies with positions balanced.

## What the owner does

1. Make a folder outside the repository, for example `~/drift-held-out`, and never paste its contents into a chat.
2. Write `names.json` there with names you invent: six streets, six coffee shops and a few clinics. For example: `{"streets": [...], "shops": [...], "clinics": [...]}`. The runner refuses a file where one name appears inside another, because that would break the scoring.
3. Write a number of your choosing into `seed.txt`. Leaving it off the command line keeps it out of shell history and process lists, and the report records only that a seed file was used.
4. When the agent says the cluster is ready, run the following from the repository with the usual host and MCDMA environment:

   ```bash
   PYTHONPATH=. .venv/bin/python scripts/live/demo_appointment.py --scenarios 24 --balance --selection all --prompt-copies 3 --reserve 1536 --conditions linked no_link --total-seconds 3600 --summary-only --seed-file ~/drift-held-out/seed.txt --vocabulary-file ~/drift-held-out/names.json --out ~/drift-held-out/run-1
   ```

5. Score it, and share only this output:

   ```bash
   PYTHONPATH=. .venv/bin/python scripts/score_reverse_recall.py ~/drift-held-out/run-1/report.json
   ```

## What stays private

The run folder holds the prompts, the models' answers and the names. With `--summary-only` the runner prints only final counts. The scorer prints counts, the sign test, per-position totals and the verdict, never a name or an answer.

## How strong the barrier is

The agent runs as the owner's user account, so a folder it is asked not to read is a promise, not an access control. For a real barrier, run steps 1 to 5 from a separate macOS account whose home folder is closed to others, or from another machine with the same host access. The result then says which barrier held it.
