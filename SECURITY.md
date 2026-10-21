# Security

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability
reporting on this repository (Security tab, "Report a vulnerability"). Include what you found,
the steps to reproduce it, and the version or commit.

You can expect an acknowledgement, and a fix or a decision, as soon as the maintainers can
manage; this is a small open-source project without a service-level commitment.

## Scope

Cutpoint runs locally and invokes ffmpeg, ffprobe and, for the optional `events` layers, a local
`llama-server` process. Relevant reports include unsafe handling of file names or paths from
media and JSON inputs (segments files, presets, `config.json`), unsafe downloads in
`cutpoint setup`, and anything that could make Cutpoint modify source videos or write outside
its project and data folders.

The Insta360 SDK is third-party software under Insta360's own terms. Report SDK problems to
Insta360.
