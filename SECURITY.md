# Security Policy

## Supported Versions

Security and reliability fixes are applied to the `main` branch and the latest minor release.

| Version | Supported |
|---|---|
| `main` / `1.x` | ✅ |

## Secret & Credential Hygiene

- **Never commit** GitHub Personal Access Tokens, API keys, `.env` files, or private host credentials to this repository.
- External service keys (if used by downstream extensions) must be supplied exclusively via environment variables at runtime.
- Model weights and private user videos must stay outside Git history (see `.gitignore`).

## Reporting a Vulnerability

If you discover a security vulnerability or accidental credential exposure:

1. **Do not** disclose exploit details in a public GitHub issue.
2. Report privately via GitHub Security Advisories or contact the repository maintainers directly.
3. If a token or credential was exposed, revoke and rotate it immediately before rewriting history.
