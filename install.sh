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
    x86_64 | amd64) target="x86_64-unknown-linux-musl" ;;
    aarch64 | arm64) target="aarch64-unknown-linux-musl" ;;
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

require_command awk
require_command curl
require_command install
require_command mktemp
require_command sha256sum
require_command tar

asset="mcode-$target.tar.gz"
temporary_dir="$(mktemp -d)"
cleanup() {
    rm -rf "$temporary_dir"
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

printf 'Downloading %s...\n' "$asset"
curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fL --retry 3 \
    -o "$temporary_dir/$asset" "$release_url/$asset"
curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fL --retry 3 \
    -o "$temporary_dir/SHA256SUMS" "$release_url/SHA256SUMS"

checksum_line="$(
    awk -v asset="$asset" '$2 == asset || $2 == "*" asset { print; found = 1 } END { if (!found) exit 1 }' \
        "$temporary_dir/SHA256SUMS"
)" || die "checksum not found for $asset"
(
    cd "$temporary_dir"
    printf '%s\n' "$checksum_line" | sha256sum -c -
) || die "checksum verification failed"

tar -xzf "$temporary_dir/$asset" -C "$temporary_dir" mcode
mkdir -p "$install_dir"
install -m 0755 "$temporary_dir/mcode" "$install_dir/mcode"

printf 'Installed %s to %s/mcode\n' "$("$install_dir/mcode" --version)" "$install_dir"
case ":${PATH:-}:" in
    *":$install_dir:"*) ;;
    *) printf 'Add %s to PATH before running mcode.\n' "$install_dir" ;;
esac
