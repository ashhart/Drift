# Shared space: adding a model

**Status:** in place and tested. Qwen joined with one command, and GLM to Qwen through the shared space answered 27 of 32 short-context questions against 26 for the direct pairwise translator ([record](../evaluation/SHARED_SPACE_MEMBER.md)).

Pairwise translators grow as N squared: GLM, Qwen and DeepSeek already need six, and each new model adds two per member. The shared space needs two maps per model. Each model maps into one hub and out of it, and any two members compose into one translator.

## The hub

The hub is GLM-5.3's MLA latents, reduced to their top 2,048 principal directions. That is the reduction inside `rows_g2q_code.npz`, the GLM-to-Qwen rows translator the drop-in work started from. GLM is the anchor. Its encoder is that projection, and its decoder is the projection back.

Every other model is a member with two linear maps (`drift/translate/hub.py`):

- the encoder takes the model's per-token cache rows, reduces them to their own principal directions, and maps them into the hub with ridge;
- the decoder maps the hub back to the model's rows with ridge, then restores the rows' spread (`drift/translate/spread.py`), because squared error shrinks it.

A translator from member A to member B is A's encoder followed by B's decoder. Both are linear, so `hub.pair` folds them into one `ridge_map` translator. The gates and drop-in tools load it exactly as they load a pairwise one.

## What a new model has to provide

A new model needs a cache adapter with two jobs.

1. **Export rows.** For each text window, write one npz named by the window id, holding:
   - `offsets`, int32 `[tokens, 2]`: each token's start and end character in the window text;
   - `x`, float16 `[tokens, width]`: the rows the model keeps in its cache for that token, every cache-bearing layer in a fixed order, taken at the canonical boundary in `ADAPTERS.md` (keys after their norm and before rotary, values unrotated, MLA latents after their norm).

   `scripts/live/studio_tap_passages.py` is the Qwen example. Its rows are 12 full-attention layers of K and V, 1,024 values per layer (`drift/translate/qwen_rows.py`).
2. **Take rows.** Append translated rows into a running cache at explicit positions, re-applying the model's own rotary, then continue decoding. `append_entries` in `drift/serving/omlx_cache.py` does this for Qwen. The model's weights stay frozen.

The windows must be the texts the anchor already read. GLM's side comes from `export_passages.py`, with the same window ids. The fit pairs a GLM token with the new model's token that ends on the same character (`drift/translate/alignment.py`); windows GLM exported with inexact offsets are skipped. The Qwen member used GLM's 1,919 training windows of about 1,400 tokens and 4 held-out windows.

Hybrid models also carry recurrent state that has no per-token rows, such as Qwen3.8's linear-attention layers. The hub does not carry that state yet. Such a receiver still needs its own state translator (`scripts/live/fit_state_translator.py`) from GLM's reduced latents, which is the hub itself, so one state map per hybrid receiver is enough.

## One command

```bash
python3 scripts/live/hub_add_model.py --make-anchor glm --rows out/rows_g2q_code.npz --out hub/glm
```

```bash
python3 scripts/live/hub_add_model.py --name newmodel --anchor hub/glm \
    --features out/newmodel_rows --anchor-features out/mix_latents \
    --val-features out/newmodel_val --val-anchor out/code_glm_val --out hub/newmodel \
    --test-from hub/glm --test "python3 scripts/live/my_gate.py --rows {translator}"
```

The first command writes the anchor once. The second fits the new member, scores both maps on the held-out windows, and saves `member.npz` and `member.json`. With `--test`, it pairs the new member with another member (`--test-from` into the new model, or `--test-to` out of it), writes the pair as `pair_<from>_to_<to>.npz`, and runs the test command with `{translator}` replaced by that path. The command's exit code and the tail of its output go into `member.json`.

`member.json` reports:

- `encoder_r2`: how well the new model's rows land where GLM's latents for the same tokens sit in the hub;
- `decoder_r2`: how well the hub predicts the new model's rows, which is the GLM-to-new-model translator's fit;
- `through_the_hub_r2`: the new model's rows carried into the hub and out as GLM latents.

R-squared only says the maps fit. Whether a receiver answers from the translated rows is what the drop-in test measures.

## The drop-in test

For Qwen as the receiver, the test is the drop-in gate with the paired translator in place of the pairwise one:

```bash
PYTHONPATH=. $OMLX_PY scripts/live/studio_qwen_state_gate.py --items out/dropin_items.json \
    --glm-latents out/dropin_latents --glm-rows {translator} --forward-state out/state.npz \
    --arms text,none,t_rows,t_rows_state --out out/hub_gate.json
```

Another receiver needs its own gate built on its adapter's second job. The gate's rules still hold: the `none` arm shows what the model answers without any memory, and a translator counts only where the rows arm beats it.

## Caches without per-token rows

Some caches keep one entry per group of tokens. DeepSeek V4 keeps one per 4 tokens in 21 layers and one per 128 in 20
more. Such a model joins as a receiver through a grouped decoder (`drift/translate/dsv4_member.py`,
`scripts/live/hub_fit_dsv4.py`). Each entry's input is the mean hub vector of the sender's tokens that end inside the
entry's characters, and ridge maps give the entry's content. Positional parts of an entry, such as rotated key
dimensions, cannot come from content; the writer keeps the model's own values for placeholder text at the same
positions. The model's adapter must then also:

- decode and encode its cache pages exactly (`drift/translate/dsv4_pages.py` round-trips DeepSeek's float8 pages);
- tap a placeholder span, so the writer has its positional values;
- write whole pages, so spans start on a page boundary.

One grouped decoder serves every sender, because each sender is already in the hub: DeepSeek's took GLM through the
anchor and Qwen through its encoder with no further fitting. How well it works is another matter: see
[the six directions](../evaluation/SIX_DIRECTIONS.md).

## Limits

- The maps are linear. On the GLM-to-Qwen pair, the contextual reader and answer-level fine-tuning in `docs/evaluation/DROPIN_REAL_PROJECT.md` carried the rows well past a linear map. Both are trained per receiver, starting from a linear translator, and a hub pair can be that start (`--rows` in `studio_train_context_reader.py`).
- Only GLM's side is fixed. A member's fit depends on how well the anchor's latents carry what the member keeps in its cache.
- Recurrent state goes through a per-receiver state map, as above.
