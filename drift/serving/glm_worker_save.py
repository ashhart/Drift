"""Save connector steps and surface rank-wide failures to the engine."""
import json

try:
    from live_receiver_glm import LiveReceiverError, failure_diagnostics, failure_message
except ImportError:
    from drift.serving.live_receiver_glm import LiveReceiverError, failure_diagnostics, failure_message


def save_step(connector, step, live, logger):
    error, report = None, None
    try:
        if step.failed:
            if getattr(step, "failed_code", ""):
                raise LiveReceiverError(step.failed_code, **dict(step.failed_details))
            raise RuntimeError(step.failed)
        if live:
            connector._live_step(step)
        else:
            report = connector._drift_write(step)
    except Exception as caught:
        error = caught
    try:
        failed = connector._drift_rank_failed(error is not None)
    except Exception:
        failed = True
        error = RuntimeError("rank failure agreement unavailable")
    if failed:
        error = error or RuntimeError("peer rank cache save failed")
        logger.error("drift cache save failed: %s", json.dumps(failure_diagnostics(error)))
        if live:
            connector._live_worker.setdefault(step.name, {})["poisoned"] = True
        try:
            if live:
                folder = connector._live_out / step.name
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"error.rank{connector._tp()[0]}").write_text(failure_message(error))
            else:
                connector._tp_report(step.name, connector._tp()[0], "error", {"error": str(error)})
        except OSError:
            pass
        raise RuntimeError("Drift cache save failed; session poisoned") from None
    if not live:
        connector._tp_report(step.name, connector._tp()[0], "done", report)
