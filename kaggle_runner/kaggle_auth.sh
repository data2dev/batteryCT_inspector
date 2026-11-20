#!/usr/bin/env bash
# Configure Kaggle CLI credentials on a GitHub runner and export KAGGLE_USER.
#
# Preferred: secret KAGGLE_API_TOKEN (Kaggle -> Settings -> API -> Generate New Token).
#            The username is looked up from the token, so no other secret is needed.
# Fallback:  legacy secrets KAGGLE_USERNAME + KAGGLE_KEY (kaggle.json).
#
# Usage: kaggle_runner/kaggle_auth.sh [--optional]
#   --optional: if no credentials are set, write SKIP=1 to $GITHUB_ENV and exit 0.
set -euo pipefail
optional="${1:-}"
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle

if [ -n "${KAGGLE_API_TOKEN:-}" ]; then
  umask 077
  printf '%s' "$KAGGLE_API_TOKEN" > ~/.kaggle/access_token
  method="API token"
elif [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
  umask 077
  printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json
  method="legacy key"
else
  if [ "$optional" = "--optional" ]; then
    echo "No Kaggle credentials set; skipping."
    echo "SKIP=1" >> "${GITHUB_ENV:-/dev/null}"
    exit 0
  fi
  echo "::error::Set repo secret KAGGLE_API_TOKEN (or legacy KAGGLE_USERNAME + KAGGLE_KEY)."
  exit 1
fi

# Do not pass the token through the environment from here on.
unset KAGGLE_API_TOKEN KAGGLE_KEY

out=$(python - <<'PYEOF' 2>&1 || true
import sys
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception as e:  # e.g. a missing dependency: not a credentials problem
    print(f"@@ERROR=load:{type(e).__name__}: {e}")
    sys.exit(0)
try:
    api = KaggleApi()
    api.authenticate()
    name = api.config_values.get("username") or ""
    api.kernels_list(mine=True, page_size=1)  # real request; legacy keys are not checked until used
except SystemExit:
    print("@@ERROR=auth:Kaggle did not accept the credentials")
    sys.exit(0)
except Exception as e:
    code = getattr(getattr(e, "response", None), "status_code", None)
    kind = "auth" if code in (401, 403) else "other"
    print(f"@@ERROR={kind}:{type(e).__name__} (HTTP {code})")
    sys.exit(0)
if name:
    print(f"@@KAGGLE_USER={name}")
else:
    print("@@ERROR=auth:no username returned for these credentials")
PYEOF
)
# Only trust explicit marker lines: the CLI prints help text on failures.
user=$(printf '%s\n' "$out" | sed -n 's/^@@KAGGLE_USER=\([A-Za-z0-9_-]\{1,64\}\)$/\1/p' | tail -n1)
if [ -n "$user" ]; then
  echo "Authenticated to Kaggle as $user ($method)."
  echo "KAGGLE_USER=$user" >> "${GITHUB_ENV:-/dev/null}"
  exit 0
fi
err=$(printf '%s\n' "$out" | sed -n 's/^@@ERROR=//p' | tail -n1)
case "$err" in
  load:*)
    echo "::error::The Kaggle CLI failed to load (${err#load:}). This is an install problem, not a credentials problem." ;;
  auth:*)
    echo "::error::Kaggle rejected the credentials ($method): ${err#auth:}. Generate a new token and update the secret." ;;
  *)
    printf '%s\n' "$out" | grep -v -i -E 'token|key' | tail -n5 >&2 || true
    echo "::error::Kaggle check failed ($method): ${err#other:}. See output above." ;;
esac
exit 1
