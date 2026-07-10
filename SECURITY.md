# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately to the repository owner using
the contact method on their GitHub profile. Do not open a public issue containing
credentials, access tokens, private model URLs, or exploit details.

## Credential hygiene

This repository does not require committed credentials. Store Hugging Face,
Google Drive, cluster, and experiment-tracking tokens in environment variables
or the provider's local credential store. Files such as `.env`, `token.json`,
`client_secrets.json`, and `config/secret_key` are ignored.

If a credential has ever been committed, removing the file in a later commit is
not sufficient: revoke or rotate it, then rewrite Git history before making the
repository public.
