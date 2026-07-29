use std::cmp::Ordering;
use std::io::{self, Read};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use futures_util::StreamExt;
use reqwest::{Client, Response};
use semver::Version;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;
use url::Url;
use uuid::Uuid;

const RELEASE_API_URL: &str = "https://api.github.com/repos/imengying/MCode/releases/latest";
const MAX_RELEASE_METADATA_BYTES: usize = 1024 * 1024;
const MAX_ARCHIVE_BYTES: usize = 64 * 1024 * 1024;
const MAX_UNPACKED_BYTES: u64 = 128 * 1024 * 1024;
const MAX_BINARY_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    assets: Vec<GitHubAsset>,
}

#[derive(Debug, Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
}

pub async fn run() -> Result<()> {
    let (target, asset_name) = release_platform()?;
    let current_version = Version::parse(crate::VERSION)
        .context("the installed MCode version is not valid semantic versioning")?;
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_mins(5))
        .https_only(true)
        .user_agent(format!("mcode/{}", crate::VERSION))
        .build()
        .context("failed to create the update client")?;

    println!("Checking for MCode updates...");
    let metadata = download_limited(
        client
            .get(RELEASE_API_URL)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28"),
        MAX_RELEASE_METADATA_BYTES,
        "release metadata",
    )
    .await?;
    let release: GitHubRelease =
        serde_json::from_slice(&metadata).context("failed to decode GitHub release metadata")?;
    let latest_version = version_from_tag(&release.tag_name)?;

    match latest_version.cmp(&current_version) {
        Ordering::Less => {
            println!(
                "Installed MCode {current_version} is newer than the latest release {latest_version}."
            );
            return Ok(());
        }
        Ordering::Equal => {
            println!("MCode {current_version} is already up to date.");
            return Ok(());
        }
        Ordering::Greater => {}
    }

    let asset = release
        .assets
        .iter()
        .find(|asset| asset.name == asset_name)
        .with_context(|| format!("release {} does not contain {asset_name}", release.tag_name))?;
    let download_url = validated_download_url(&asset.browser_download_url)?;
    println!("Downloading MCode {latest_version} for {target}...");
    let compressed = download_limited(
        client.get(download_url),
        MAX_ARCHIVE_BYTES,
        "release archive",
    )
    .await?;
    let binary = extract_binary(&compressed)?;
    let executable = std::env::current_exe().context("failed to locate the MCode executable")?;
    replace_executable(&executable, &binary).await?;

    println!("Updated MCode {current_version} to {latest_version}.");
    Ok(())
}

fn release_platform() -> Result<(&'static str, &'static str)> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Ok(("x86_64-unknown-linux-musl", "MCode-amd64.tar.gz")),
        ("linux", "aarch64") => Ok(("aarch64-unknown-linux-musl", "MCode-arm64.tar.gz")),
        (os, arch) => bail!("mcode update is not supported on {os}/{arch}"),
    }
}

fn version_from_tag(tag: &str) -> Result<Version> {
    Version::parse(tag.strip_prefix('v').unwrap_or(tag))
        .context("the latest GitHub release tag is not valid semantic versioning")
}

fn validated_download_url(raw: &str) -> Result<Url> {
    let url = Url::parse(raw).context("the release asset has an invalid download URL")?;
    if url.scheme() != "https" || url.host_str() != Some("github.com") {
        bail!("the release asset download URL is not an official GitHub HTTPS URL");
    }
    Ok(url)
}

async fn download_limited(
    request: reqwest::RequestBuilder,
    limit: usize,
    description: &str,
) -> Result<Vec<u8>> {
    let response = request
        .send()
        .await
        .with_context(|| format!("failed to request {description}"))?
        .error_for_status()
        .with_context(|| format!("failed to download {description}"))?;
    response_bytes_limited(response, limit, description).await
}

async fn response_bytes_limited(
    response: Response,
    limit: usize,
    description: &str,
) -> Result<Vec<u8>> {
    let limit_u64 = u64::try_from(limit).unwrap_or(u64::MAX);
    if response
        .content_length()
        .is_some_and(|length| length > limit_u64)
    {
        bail!("{description} exceeds the download limit");
    }
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("failed to read {description}"))?;
        if bytes.len().saturating_add(chunk.len()) > limit {
            bail!("{description} exceeds the download limit");
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

fn extract_binary(compressed: &[u8]) -> Result<Vec<u8>> {
    let decoder = GzDecoder::new(compressed);
    let mut archive = tar::Archive::new(decoder);
    let mut binary = None;
    let mut unpacked_bytes = 0_u64;
    {
        let entries = archive
            .entries()
            .context("failed to read the release archive")?;
        for entry in entries {
            let mut entry = entry.context("failed to read an entry in the release archive")?;
            let entry_size = entry
                .header()
                .size()
                .context("release archive contains an invalid entry size")?;
            unpacked_bytes = unpacked_bytes
                .checked_add(entry_size)
                .context("release archive size overflow")?;
            if unpacked_bytes > MAX_UNPACKED_BYTES {
                bail!("release archive exceeds the unpacked size limit");
            }
            if entry.path()?.as_ref() != Path::new("mcode") {
                continue;
            }
            if binary.is_some() {
                bail!("release archive contains more than one mcode binary");
            }
            if !entry.header().entry_type().is_file() {
                bail!("the mcode archive entry is not a regular file");
            }
            if entry_size > MAX_BINARY_BYTES {
                bail!("the mcode binary exceeds the size limit");
            }
            let capacity = usize::try_from(entry_size).context("mcode binary is too large")?;
            let mut contents = Vec::with_capacity(capacity);
            entry
                .read_to_end(&mut contents)
                .context("failed to extract the mcode binary")?;
            binary = Some(contents);
        }
    }
    let mut decoder = archive.into_inner();
    io::copy(&mut decoder, &mut io::sink()).context("release archive checksum is invalid")?;

    let binary = binary.context("release archive does not contain the mcode binary")?;
    if !binary.starts_with(b"\x7fELF") {
        bail!("the release archive does not contain a Linux executable");
    }
    Ok(binary)
}

async fn replace_executable(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("executable path has no parent: {}", path.display()))?;
    let temporary = parent.join(format!(".mcode-update-{}.tmp", Uuid::now_v7()));
    let result = async {
        let mut file = tokio::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .await
            .with_context(|| {
                format!(
                    "failed to create an update file beside {}; is the directory writable?",
                    path.display()
                )
            })?;
        file.write_all(contents)
            .await
            .context("failed to write the updated executable")?;
        file.flush()
            .await
            .context("failed to flush the updated executable")?;
        file.sync_all()
            .await
            .context("failed to sync the updated executable")?;
        drop(file);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            tokio::fs::set_permissions(&temporary, std::fs::Permissions::from_mode(0o755))
                .await
                .context("failed to make the updated executable runnable")?;
        }
        tokio::fs::rename(&temporary, path)
            .await
            .with_context(|| format!("failed to replace {}; is it writable?", path.display()))?;
        Ok(())
    }
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&temporary).await;
    }
    result
}

#[cfg(test)]
mod tests {
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn parses_release_versions() {
        assert_eq!(version_from_tag("v1.2.3").unwrap(), Version::new(1, 2, 3));
        assert!(version_from_tag("latest").is_err());
    }

    #[test]
    fn uses_the_published_archive_name() {
        let (_, asset_name) = release_platform().unwrap();
        let expected = match std::env::consts::ARCH {
            "x86_64" => "MCode-amd64.tar.gz",
            "aarch64" => "MCode-arm64.tar.gz",
            architecture => panic!("unsupported test architecture: {architecture}"),
        };
        assert_eq!(asset_name, expected);
    }

    #[test]
    fn accepts_only_official_asset_urls() {
        assert!(
            validated_download_url(
                "https://github.com/imengying/MCode/releases/download/v1.2.3/mcode.tar.gz"
            )
            .is_ok()
        );
        assert!(validated_download_url("http://github.com/example").is_err());
        assert!(validated_download_url("https://example.com/mcode.tar.gz").is_err());
    }

    #[test]
    fn extracts_the_linux_binary() {
        let expected = b"\x7fELFfixture";
        let archive = release_archive(expected);
        assert_eq!(extract_binary(&archive).unwrap(), expected);
    }

    #[test]
    fn rejects_non_executable_archive_content() {
        let archive = release_archive(b"not an ELF binary");
        assert!(extract_binary(&archive).is_err());
    }

    #[test]
    fn rejects_a_corrupted_release_archive() {
        let mut archive = release_archive(b"\x7fELFfixture");
        let last = archive.last_mut().unwrap();
        *last ^= 0xff;
        assert!(extract_binary(&archive).is_err());
    }

    #[tokio::test]
    async fn atomically_replaces_the_executable() {
        let directory = tempdir().unwrap();
        let executable = directory.path().join("mcode");
        std::fs::write(&executable, b"old").unwrap();

        replace_executable(&executable, b"\x7fELFnew")
            .await
            .unwrap();

        assert_eq!(std::fs::read(&executable).unwrap(), b"\x7fELFnew");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            assert_eq!(
                std::fs::metadata(&executable).unwrap().permissions().mode() & 0o777,
                0o755
            );
        }
    }

    fn release_archive(binary: &[u8]) -> Vec<u8> {
        let encoder = GzEncoder::new(Vec::new(), Compression::default());
        let mut archive = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_path("mcode").unwrap();
        header.set_size(u64::try_from(binary.len()).unwrap());
        header.set_mode(0o755);
        header.set_cksum();
        archive.append(&header, binary).unwrap();
        let encoder = archive.into_inner().unwrap();
        encoder.finish().unwrap()
    }
}
