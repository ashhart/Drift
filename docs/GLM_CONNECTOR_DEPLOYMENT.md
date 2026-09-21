# GLM connector deployment preparation

The checked-in manifest pins the local connector dependency closure but contains
example targets, not a live deployment inventory. Its zero runtime hashes and empty
rollback observations must be replaced with verified private measurements before
staging. Host aliases, container names and runtime paths are operator configuration.
Keep the completed deployment manifest and receipts outside the repository.

## Prepare and inspect

The connector is deployed as a complete flat module bundle. The entry point maps
`drift/serving/vllm_glm53_inject.py` to `drift_glm53_connector.py`; local helper imports
are discovered recursively. Source mismatches, missing helpers, unsupported imports
and malformed target identities are refused. The two target hosts must be distinct.

Start with a private copy of `configs/glm_connector_deployment.json`, configure the
two hosts, and verify their runtime image, Python version, base connector and active
file hashes. A placeholder or stale identity must not be treated as qualification.

```sh
python -m scripts.deploy_glm pin --manifest /private/drift/deployment.json --output /private/drift/reviewed.json
python -m scripts.deploy_glm prepare --manifest /private/drift/reviewed.json --output /private/drift/candidate
```

`pin` records the checkout commit and exact current source hashes. `prepare` validates
the closure and writes the candidate, manifest and rollback inventory into a new
local directory. Neither operation copies files to a server or restarts a process.
Existing output paths are refused. Source pins do not establish native compatibility.

After host ownership and read-only inspection are authorized, `pin --refresh-rollback`
can record the actual active files for every module. Plain `pin` preserves the supplied
inventory; it cannot turn missing observations into verified rollback evidence.

An authorized operator may then copy the whole reviewed candidate to an inactive
directory with the same absolute path in both containers and request staged checks:

```sh
python -m scripts.deploy_glm check-staged --manifest /private/drift/reviewed.json --candidate /opt/drift/inactive-candidate
```

The checker verifies image identity, Python version, base connector and all candidate
hashes before importing. Every helper is imported in a fresh process without writing
bytecode; origins, inheritance, unchanged hashes and the active rollback inventory
must agree. Unlisted files or directories, fallback imports, changed runtime identities,
timeouts or duplicate host receipts keep the aggregate BLOCKED. Both targets are checked
even if one fails. SSH operations have bounded connection and subprocess deadlines.

## Cutover is separate

PASSED staged imports do not prove native requests or that a serving process has
reloaded code. Preserve verified rollback files and launch configuration privately,
obtain both staged receipts, and authorize a coordinated maintenance window before
activation. Both ranks must switch to the same complete bundle. Never overlay helper
files incrementally or resume one rank after a partial promotion.

This tool implements no deployment, restart or rollback mutation. A bounded native
qualification still follows any coordinated cutover. Local synthetic tests cover
source drift, dependency closure, both-host admission, receipt identity, failure handling
and fresh-process import provenance; they do not qualify real models or advance M4/M5.
