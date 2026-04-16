# Third-Party References

This project expects local clones of the upstream reference repositories in this directory:

- `third_party/Trojan-Activation-Attack`
- `third_party/camel`

Use `scripts/setup_references.sh` to clone or update them.

By default the script checks out the exact commits listed in `third_party/refs.lock`.
That gives you a deterministic and auditable third-party state instead of tracking
whatever happens to be on the upstream default branches at runtime.

The experiment code prepends `third_party/camel` to `sys.path` when present so local CAMEL source is used ahead of the pip-installed package.
