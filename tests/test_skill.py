"""``cutpoint skill install | print`` and the packaged SKILL.md (frontmatter, CLI drift guard)."""

from __future__ import annotations

import argparse
import re
import shlex

import pytest

from cutpoint import cli, skill


def _skill_text() -> str:
    return skill.skill_bytes().decode("utf-8")


def test_install_writes_skill_md(tmp_path, caplog):
    caplog.set_level("INFO", logger="cutpoint")
    assert cli.main(["skill", "install", "--dir", str(tmp_path / "skills")]) == 0
    target = tmp_path / "skills" / "cutpoint" / "SKILL.md"
    assert target.read_bytes() == skill.skill_bytes()
    assert str(target) in caplog.text
    assert list(target.parent.iterdir()) == [target]  # no temp file left


def test_install_default_dir_is_user_claude_skills(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile"))
    assert cli.main(["skill", "install"]) == 0
    target = tmp_path / "profile" / ".claude" / "skills" / "cutpoint" / "SKILL.md"
    assert target.read_bytes() == skill.skill_bytes()


def test_identical_reinstall_is_a_noop(tmp_path, monkeypatch):
    skills = tmp_path / "skills"
    assert cli.main(["skill", "install", "--dir", str(skills)]) == 0
    target = skills / "cutpoint" / "SKILL.md"

    def no_write(*args, **kwargs):
        raise AssertionError("identical content must not be rewritten")

    monkeypatch.setattr(skill, "save_bytes", no_write)
    assert cli.main(["skill", "install", "--dir", str(skills)]) == 0
    assert target.read_bytes() == skill.skill_bytes()


def test_differing_target_without_force_fails_and_keeps_file(tmp_path, caplog):
    target = tmp_path / "skills" / "cutpoint" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"my own edits\n")
    assert cli.main(["skill", "install", "--dir", str(tmp_path / "skills")]) == 1
    assert target.read_bytes() == b"my own edits\n"
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert errors
    assert str(target) in errors[0].getMessage()


def test_force_replaces_differing_target(tmp_path):
    target = tmp_path / "skills" / "cutpoint" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old\n")
    assert cli.main(["skill", "install", "--dir", str(tmp_path / "skills"), "--force"]) == 0
    assert target.read_bytes() == skill.skill_bytes()


def test_print_writes_skill_to_stdout(capsysbinary):
    assert cli.main(["skill", "print"]) == 0
    assert capsysbinary.readouterr().out == skill.skill_bytes()


def test_skill_without_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        cli.main(["skill"])
    assert exc.value.code == 2


def test_frontmatter_has_name_and_description():
    text = _skill_text()
    match = re.match(r"---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    assert match, "SKILL.md must start with YAML frontmatter"
    fields = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        assert sep, f"not a 'key: value' line: {line!r}"
        fields[key.strip()] = value.strip()
    assert fields["name"] == "cutpoint"
    assert fields["description"]


# --- drift guard: every command and option the skill names exists in the real parser ---


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _all_options(parser: argparse.ArgumentParser) -> set[str]:
    found = set(parser._option_string_actions)
    for child in _subparsers(parser).values():
        found |= _all_options(child)
    return found


def _commands(text: str) -> list[list[str]]:
    """Token lists after ``cutpoint`` in every code span and fenced line of the skill."""
    spans = re.findall(r"`([^`\n]+)`", text)
    spans += re.findall(r"```[a-z]*\n(.*?)```", text, re.S)
    found = []
    for span in spans:
        for line in span.splitlines():
            m = re.search(r"\bcutpoint\s+(.*)", line)
            if m and not line.lstrip().startswith(("name:", "description:")):
                found.append(shlex.split(m.group(1).replace("|", " "), posix=False))
    return found


def test_skill_commands_and_options_exist_in_parser():
    root = cli.build_parser()
    commands = _commands(_skill_text())
    assert len(commands) > 10, "the parser of the skill text found too few commands"
    for tokens in commands:
        parser = root
        path = []
        for tok in tokens:  # descend through subcommands
            children = _subparsers(parser)
            if not children or tok.startswith(("-", "<")):
                break
            assert tok in children, f"skill names unknown command: cutpoint {' '.join(path)} {tok}"
            parser = children[tok]
            path.append(tok)
        for tok in tokens:
            if tok.startswith("--"):
                opt = tok.split("=", 1)[0]
                assert opt in parser._option_string_actions or opt in root._option_string_actions, (
                    f"skill names unknown option: cutpoint {' '.join(path)} {opt}"
                )


def test_every_option_in_skill_exists_in_parser():
    known = _all_options(cli.build_parser())
    mentioned = set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", _skill_text()))
    assert mentioned
    assert mentioned <= known, f"unknown options: {sorted(mentioned - known)}"
