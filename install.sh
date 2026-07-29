#!/bin/sh
set -eu

REPOSITORY="${MCODE_REPOSITORY:-imengying/MCode}"
VERSION="${MCODE_VERSION:-latest}"

die() {
    printf 'mcode installer: %s\n' "$1" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

[ "$(uname -s)" = "Linux" ] || die "only Linux is supported"

case "$(uname -m)" in
    x86_64 | amd64) asset="MCode-amd64.tar.gz" ;;
    aarch64 | arm64) asset="MCode-arm64.tar.gz" ;;
    *) die "unsupported architecture: $(uname -m)" ;;
esac

if [ -n "${MCODE_INSTALL_DIR:-}" ]; then
    install_dir="$MCODE_INSTALL_DIR"
elif [ -n "${HOME:-}" ]; then
    install_dir="$HOME/.local/bin"
else
    die "HOME is not set; provide MCODE_INSTALL_DIR"
fi

case "$VERSION" in
    latest) release_url="https://github.com/$REPOSITORY/releases/latest/download" ;;
    v*) release_url="https://github.com/$REPOSITORY/releases/download/$VERSION" ;;
    *) release_url="https://github.com/$REPOSITORY/releases/download/v$VERSION" ;;
esac

require_command curl
require_command install
require_command mktemp
require_command tar

temporary_dir="$(mktemp -d)"
cleanup() {
    rm -rf "$temporary_dir"
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

printf 'Downloading %s...\n' "$asset"
curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fL --retry 3 \
    -o "$temporary_dir/$asset" "$release_url/$asset"

tar -xzf "$temporary_dir/$asset" -C "$temporary_dir" mcode
mkdir -p "$install_dir"
install -m 0755 "$temporary_dir/mcode" "$install_dir/mcode"

printf 'Installed %s to %s/mcode\n' "$("$install_dir/mcode" --version)" "$install_dir"
case ":${PATH:-}:" in
    *":$install_dir:"*) ;;
    *) printf 'Add %s to PATH before running mcode.\n' "$install_dir" ;;
esac
