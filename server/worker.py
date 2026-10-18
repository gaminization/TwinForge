"""
Final-build worker.  Runs in its OWN PROCESS so that a crash or an out-of-memory kill of the heavy
reconstruction (COLMAP + dense stereo) can never take the web server down with it.

    python -m server.worker <job_id> <single|articulated> <robot_name> <size_m|none>

Protocol: progress and the final state are printed as lines starting with '@@' + JSON.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def emit(obj):
    print("@@" + json.dumps(obj, default=str), flush=True)


def main():
    jid, mode, name, size = sys.argv[1:5]
    size_m = None if size == "none" else float(size)
    from server import app as A
    A._jobs[jid] = {"status": "running"}
    A._loop = None

    def prog(jid_, pct, stage, msg):
        A._jobs[jid].update({"pct": pct, "stage": stage, "msg": msg})
        emit({"t": "prog", "pct": pct, "stage": stage, "msg": msg})
    A._prog = prog
    (A._run_articulated_sync if mode == "articulated" else A._run_sync)(jid, name, size_m)
    emit({"t": "final", "job": A._jobs[jid]})


if __name__ == "__main__":
    main()
