use std::env;
use std::path::Path;
use std::process::Stdio;

use anyhow::{Result, bail};
use tokio::process::Command;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PermissionProfile {
    ReadOnly,
    #[default]
    WorkspaceWrite,
    FullAccess,
}

impl PermissionProfile {
    pub const ALL: [Self; 3] = [Self::ReadOnly, Self::WorkspaceWrite, Self::FullAccess];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ReadOnly => "只读",
            Self::WorkspaceWrite => "工作区可写",
            Self::FullAccess => "完全访问",
        }
    }

    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::ReadOnly => "工作区只读，/tmp 可写，禁用网络",
            Self::WorkspaceWrite => "仅可写当前项目和 /tmp，禁用网络",
            Self::FullAccess => "不使用系统沙箱，命令仍需审批",
        }
    }

    #[must_use]
    pub const fn allows_file_writes(self) -> bool {
        !matches!(self, Self::ReadOnly)
    }
}

pub(crate) fn shell_command(
    command: &str,
    root: &Path,
    profile: PermissionProfile,
) -> Result<Command> {
    let shell = env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
    let mut process = if profile == PermissionProfile::FullAccess {
        let mut process = Command::new(shell);
        process.arg("-lc").arg(command);
        process
    } else {
        ensure_bubblewrap_available()?;
        let mut process = Command::new("bwrap");
        process
            .arg("--die-with-parent")
            .arg("--new-session")
            .arg("--unshare-net")
            .arg("--ro-bind")
            .arg("/")
            .arg("/")
            .arg("--proc")
            .arg("/proc")
            .arg("--dev")
            .arg("/dev")
            .arg("--bind")
            .arg("/tmp")
            .arg("/tmp");
        if profile == PermissionProfile::WorkspaceWrite {
            process.arg("--bind").arg(root).arg(root);
        }
        process
            .arg("--chdir")
            .arg(root)
            .arg("--")
            .arg(shell)
            .arg("-lc")
            .arg(command);
        process
    };
    process
        .current_dir(root)
        .env("AI_AGENT", "mcode")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    Ok(process)
}

fn ensure_bubblewrap_available() -> Result<()> {
    let path = env::var_os("PATH").unwrap_or_default();
    if env::split_paths(&path).any(|directory| directory.join("bwrap").is_file()) {
        return Ok(());
    }
    bail!(
        "当前权限档位需要 Bubblewrap（bwrap）；请安装 bubblewrap，或通过 /permissions 选择完全访问"
    )
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn full_access_shell_does_not_use_bubblewrap() {
        let project = tempdir().unwrap();
        let command =
            shell_command("printf ok", project.path(), PermissionProfile::FullAccess).unwrap();

        assert_ne!(command.as_std().get_program(), "bwrap");
        assert!(
            command
                .as_std()
                .get_args()
                .any(|argument| argument == "printf ok")
        );
    }
}
