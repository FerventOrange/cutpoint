"""``cutpoint skill install | print`` and the packaged SKILL.md (frontmatter, CLI drift guard)."""

from __future__ import annotations

import argparse
import re
import shlex

import pytest

from cutpoint import cli, skill
from cutpoint.edit import KINDS
from cutpoint.styles import STYLES


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
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile"))
    assert cli.main(["skill", "install"]) == 0
    target = tmp_path / "profile" / ".claude" / "skills" / "cutpoint" / "SKILL.md"
    assert target.read_bytes() == skill.skill_bytes()


def test_install_default_dir_honours_claude_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    assert cli.main(["skill", "install"]) == 0
    target = tmp_path / "claude-config" / "skills" / "cutpoint" / "SKILL.md"
    assert target.read_bytes() == skill.skill_bytes()
    assert not (tmp_path / "profile" / ".claude").exists()


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
                found.append(shlex.split(m.group(1), posix=False))
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
        for i, tok in enumerate(tokens):
            if not tok.startswith("--"):
                continue
            opt = tok.split("=", 1)[0]
            action = parser._option_string_actions.get(opt) or root._option_string_actions.get(opt)
            assert action, f"skill names unknown option: cutpoint {' '.join(path)} {opt}"
            # values of an option with choices: `--kind a|b` names each choice, `<x>` is a slot
            value = tok.partition("=")[2] or (tokens[i + 1] if i + 1 < len(tokens) else "")
            if action.choices and value and not value.startswith(("-", "<")):
                for choice in value.split("|"):
                    assert choice in action.choices, (
                        f"skill names unknown value: cutpoint {' '.join(path)} {opt} {choice}"
                    )


def test_skill_lists_every_kind():
    lists = [
        tokens[tokens.index("--kind") + 1]
        for tokens in _commands(_skill_text())
        if tokens[:1] == ["edit"] and "--kind" in tokens[:-1]
    ]
    assert lists, "the skill shows no `cutpoint edit ... --kind a|b|...`"
    for values in lists:
        assert set(values.split("|")) == set(KINDS)


def test_skill_style_list_matches_styles():
    text = " ".join(_skill_text().split())
    m = re.search(r"Style \(`--style`.*?Pick (.*?); anything else", text)
    assert m, "the skill's style sentence is missing"
    assert set(re.findall(r"`([^`]+)`", m.group(1))) == set(STYLES)


def test_every_option_in_skill_exists_in_parser():
    known = _all_options(cli.build_parser())
    mentioned = set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", _skill_text()))
    assert mentioned
    assert mentioned <= known, f"unknown options: {sorted(mentioned - known)}"
