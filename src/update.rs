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
    let (platform, asset_name) = release_platform()?;
    let current_version =
        Version::parse(crate::VERSION).context("已安装的 MCode 版本不是有效的语义化版本")?;
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_mins(5))
        .https_only(true)
        .user_agent(format!("mcode/{}", crate::VERSION))
        .build()
        .context("创建更新客户端失败")?;

    println!("正在检查 MCode 更新...");
    let metadata = download_limited(
        client
            .get(RELEASE_API_URL)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28"),
        MAX_RELEASE_METADATA_BYTES,
        "Release 元数据",
    )
    .await?;
    let release: GitHubRelease =
        serde_json::from_slice(&metadata).context("解析 GitHub Release 元数据失败")?;
    let latest_version = version_from_tag(&release.tag_name)?;

    match latest_version.cmp(&current_version) {
        Ordering::Less => {
            println!("已安装的 MCode {current_version} 比最新 Release {latest_version} 更新。");
            return Ok(());
        }
        Ordering::Equal => {
            println!("MCode {current_version} 已是最新版本。");
            return Ok(());
        }
        Ordering::Greater => {}
    }

    let asset = release
        .assets
        .iter()
        .find(|asset| asset.name == asset_name)
        .with_context(|| format!("Release {} 不包含 {asset_name}", release.tag_name))?;
    let download_url = validated_download_url(&asset.browser_download_url)?;
    println!("正在下载适用于 {platform} 的 MCode {latest_version}...");
    let compressed = download_limited(
        client.get(download_url),
        MAX_ARCHIVE_BYTES,
        "Release 压缩包",
    )
    .await?;
    let binary = extract_binary(&compressed)?;
    let executable = std::env::current_exe().context("无法定位 MCode 可执行文件")?;
    replace_executable(&executable, &binary).await?;

    println!("MCode 已从 {current_version} 更新到 {latest_version}。");
    Ok(())
}

fn release_platform() -> Result<(&'static str, &'static str)> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Ok(("linux-amd64", "MCode-amd64.tar.gz")),
        ("linux", "aarch64") => Ok(("linux-arm64", "MCode-arm64.tar.gz")),
        (os, arch) => bail!("mcode update 不支持 {os}/{arch}"),
    }
}

fn version_from_tag(tag: &str) -> Result<Version> {
    Version::parse(tag.strip_prefix('v').unwrap_or(tag))
        .context("最新 GitHub Release 标签不是有效的语义化版本")
}

fn validated_download_url(raw: &str) -> Result<Url> {
    let url = Url::parse(raw).context("Release 产物的下载 URL 无效")?;
    if url.scheme() != "https" || url.host_str() != Some("github.com") {
        bail!("Release 产物下载 URL 不是 GitHub 官方 HTTPS URL");
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
        .with_context(|| format!("请求{description}失败"))?
        .error_for_status()
        .with_context(|| format!("下载{description}失败"))?;
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
        bail!("{description}超过下载大小限制");
    }
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("读取{description}失败"))?;
        if bytes.len().saturating_add(chunk.len()) > limit {
            bail!("{description}超过下载大小限制");
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
        let entries = archive.entries().context("读取 Release 压缩包失败")?;
        for entry in entries {
            let mut entry = entry.context("读取 Release 压缩包条目失败")?;
            let entry_size = entry
                .header()
                .size()
                .context("Release 压缩包包含大小无效的条目")?;
            unpacked_bytes = unpacked_bytes
                .checked_add(entry_size)
                .context("Release 压缩包大小溢出")?;
            if unpacked_bytes > MAX_UNPACKED_BYTES {
                bail!("Release 压缩包超过解压大小限制");
            }
            if entry.path()?.as_ref() != Path::new("mcode") {
                continue;
            }
            if binary.is_some() {
                bail!("Release 压缩包包含多个 mcode 二进制文件");
            }
            if !entry.header().entry_type().is_file() {
                bail!("压缩包中的 mcode 条目不是普通文件");
            }
            if entry_size > MAX_BINARY_BYTES {
                bail!("mcode 二进制文件超过大小限制");
            }
            let capacity = usize::try_from(entry_size).context("mcode 二进制文件过大")?;
            let mut contents = Vec::with_capacity(capacity);
            entry
                .read_to_end(&mut contents)
                .context("提取 mcode 二进制文件失败")?;
            binary = Some(contents);
        }
    }
    let mut decoder = archive.into_inner();
    io::copy(&mut decoder, &mut io::sink()).context("Release 压缩包校验失败")?;

    let binary = binary.context("Release 压缩包不包含 mcode 二进制文件")?;
    if !binary.starts_with(b"\x7fELF") {
        bail!("Release 压缩包不包含 Linux 可执行文件");
    }
    Ok(binary)
}

async fn replace_executable(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("可执行文件路径没有父目录：{}", path.display()))?;
    let temporary = parent.join(format!(".mcode-update-{}.tmp", Uuid::now_v7()));
    let result = async {
        let mut file = tokio::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .await
            .with_context(|| {
                format!("无法在 {} 旁创建更新文件；该目录是否可写？", path.display())
            })?;
        file.write_all(contents)
            .await
            .context("写入新版可执行文件失败")?;
        file.flush().await.context("刷新新版可执行文件失败")?;
        file.sync_all().await.context("同步新版可执行文件失败")?;
        drop(file);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            tokio::fs::set_permissions(&temporary, std::fs::Permissions::from_mode(0o755))
                .await
                .context("设置新版可执行文件权限失败")?;
        }
        tokio::fs::rename(&temporary, path)
            .await
            .with_context(|| format!("替换 {} 失败；该文件是否可写？", path.display()))?;
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
    fn uses_the_published_platform_and_archive_names() {
        let (platform, asset_name) = release_platform().unwrap();
        let (expected_platform, expected_asset_name) = match std::env::consts::ARCH {
            "x86_64" => ("linux-amd64", "MCode-amd64.tar.gz"),
            "aarch64" => ("linux-arm64", "MCode-arm64.tar.gz"),
            architecture => panic!("unsupported test architecture: {architecture}"),
        };
        assert_eq!(platform, expected_platform);
        assert_eq!(asset_name, expected_asset_name);
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
