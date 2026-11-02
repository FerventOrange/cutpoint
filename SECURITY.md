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

`cutpoint proxy` downloads and runs a pinned, checksum-verified third-party binary
(CLIProxyAPI) bound to loopback with a per-start TLS identity, and stores subscription
credentials in the OS keyring with an encrypted recovery journal. Reports about that download,
credential storage or cleanup, the loopback connection, and API-key handling in `cutpoint ui`
are in scope.

The Insta360 SDK is third-party software under Insta360's own terms. Report SDK problems to
Insta360.
