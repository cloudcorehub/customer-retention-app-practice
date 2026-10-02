from subprocess import CompletedProcess

from scripts import preflight


def test_command_requires_nonempty_stdout(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda _: "/usr/local/bin/docker")
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args[0], 0, stdout="\n", stderr="Cannot connect to the Docker daemon"
        ),
    )

    assert not preflight.check_command(
        "Docker engine",
        ["docker", "info"],
        require_output=True,
    )


def test_command_accepts_nonempty_stdout(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda _: "/usr/bin/git")
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args[0], 0, stdout="git version 2.44.0\n", stderr=""
        ),
    )

    assert preflight.check_command("Git", ["git", "--version"], require_output=True)
