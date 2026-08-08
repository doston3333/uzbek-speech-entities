# Security Policy

## Supported versions

This project is under active development at `0.1.x`. Security fixes are applied
to the latest published revision on the default branch.

## Reporting a vulnerability

Please **do not** file public GitHub issues for security vulnerabilities,
exposed secrets, or private evaluation data.

Preferred reporting path once the repository is published on GitHub:

1. Use [GitHub Security Advisories](https://docs.github.com/en/code-security/security-advisories)
   (Security → Report a vulnerability) on this repository.
2. If advisories are not yet enabled, email the maintainer at
   **dostonbps@gmail.com** with a private description of the issue and steps to
   reproduce.

Include:

- Affected component or file paths
- Impact (for example: local file disclosure, unsafe HTML rendering, secret leak)
- Reproduction steps or a minimal proof of concept
- Whether the issue involves private audio, transcripts, or credentials

You should receive an acknowledgement within a few days. Please give the
maintainer reasonable time to investigate and ship a fix before any public
disclosure.

## Secrets and private data

- Never commit `.env` files, Modal tokens, Hugging Face tokens, or private
  recording corpora.
- Private evaluation audio and transcript-bearing results stay under
  Git-ignored paths such as `data/private_test/`.
- Diagnostic transcript logging is opt-in; leave it disabled in shared
  environments.
