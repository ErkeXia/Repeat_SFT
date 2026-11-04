"""Single-writer subprocess orchestration with inherited Linux locks and graceful stop."""
import os
from pathlib import Path
import signal
import subprocess

LOCK_ENV = "REPEAT_SFT_PIPELINE_LOCK_FD"


class Pipeline:
    def __init__(self, directory, inherited=False):
        self.directory = Path(directory)
        self.inherited = inherited
        self.child = None
        self.stop_signal = None

    def stop(self, signum, _frame):
        if self.stop_signal is None:
            self.stop_signal = signum
            print("Stopping current stage; completed results will be retained.", flush=True)
            if self.child is not None and self.child.poll() is None:
                self.child.terminate()

    def check_stop(self):
        if self.stop_signal is not None:
            raise SystemExit(128 + self.stop_signal)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file = None
        if self.inherited:
            self.fd = int(os.environ[LOCK_ENV]) if os.name == "posix" else None
            if self.fd is not None:
                os.fstat(self.fd)
        else:
            self.file = (self.directory / "pipeline.lock").open("a+b")
            try:
                if os.name == "posix":
                    import fcntl
                    fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    import msvcrt
                    if self.file.seek(0, 2) == 0:
                        self.file.write(b"\0")
                        self.file.flush()
                    self.file.seek(0)
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                self.file.close()
                raise RuntimeError(f"Job already running: {self.directory}") from exc
            self.fd = self.file.fileno()
        self.handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        for s in self.handlers:
            signal.signal(s, self.stop)
        return self

    def __exit__(self, *_):
        for s, handler in self.handlers.items():
            signal.signal(s, handler)
        if self.file is not None:
            if os.name != "posix":
                import msvcrt
                self.file.seek(0)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            self.file.close()

    def invoke(self, command, cwd):
        self.check_stop()
        options = ({"pass_fds": (self.fd,), "env": {**os.environ, LOCK_ENV: str(self.fd)}}
                   if os.name == "posix" else {})
        self.child = subprocess.Popen([str(x) for x in command], cwd=cwd, **options)
        try:
            if self.stop_signal is not None:
                self.child.terminate()
            result = self.child.wait()
        finally:
            self.child = None
        self.check_stop()
        if result:
            raise RuntimeError(f"Stage exited {result}; restart the same job to resume")
