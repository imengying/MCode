use std::sync::OnceLock;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use syntect::easy::HighlightLines;
use syntect::highlighting::{FontStyle, Style as SyntectStyle, Theme};
use syntect::parsing::{SyntaxReference, SyntaxSet};
use syntect::util::LinesWithEndings;
use two_face::theme::EmbeddedThemeName;

const MAX_HIGHLIGHT_BYTES: usize = 512 * 1024;
const MAX_HIGHLIGHT_LINES: usize = 10_000;
const MAX_HIGHLIGHT_LINE_BYTES: usize = 4 * 1024;

static SYNTAX_SET: OnceLock<SyntaxSet> = OnceLock::new();
static THEME: OnceLock<Theme> = OnceLock::new();

fn syntax_set() -> &'static SyntaxSet {
    SYNTAX_SET.get_or_init(two_face::syntax::extra_newlines)
}

fn theme() -> &'static Theme {
    THEME.get_or_init(|| {
        two_face::theme::extra()
            .get(EmbeddedThemeName::CatppuccinMocha)
            .clone()
    })
}

fn find_syntax(language: &str) -> Option<&'static SyntaxReference> {
    let syntax_set = syntax_set();
    let normalized = language.to_ascii_lowercase();
    let language = match normalized.as_str() {
        "csharp" | "c-sharp" => "c#",
        "cu" | "cuh" | "cppm" | "cxxm" | "ixx" => "cpp",
        "golang" => "go",
        "python3" => "python",
        "shell" | "sh" | "zsh" => "bash",
        _ => language,
    };

    syntax_set
        .find_syntax_by_token(language)
        .or_else(|| syntax_set.find_syntax_by_name(language))
        .or_else(|| {
            let lowercase = language.to_ascii_lowercase();
            syntax_set
                .syntaxes()
                .iter()
                .find(|syntax| syntax.name.to_ascii_lowercase() == lowercase)
        })
        .or_else(|| syntax_set.find_syntax_by_extension(language))
}

fn style_from_syntect(style: SyntectStyle) -> Style {
    let mut result = Style::default().fg(Color::Rgb(
        style.foreground.r,
        style.foreground.g,
        style.foreground.b,
    ));
    if style.font_style.contains(FontStyle::BOLD) {
        result = result.add_modifier(Modifier::BOLD);
    }
    result
}

fn highlighted_lines(code: &str, language: &str) -> Option<Vec<Line<'static>>> {
    if code.is_empty()
        || code.len() > MAX_HIGHLIGHT_BYTES
        || code.lines().count() > MAX_HIGHLIGHT_LINES
        || code
            .lines()
            .any(|line| line.len() > MAX_HIGHLIGHT_LINE_BYTES)
    {
        return None;
    }

    let syntax = find_syntax(language)?;
    let mut highlighter = HighlightLines::new(syntax, theme());
    let mut lines = Vec::new();
    for line in LinesWithEndings::from(code) {
        let ranges = highlighter.highlight_line(line, syntax_set()).ok()?;
        let spans = ranges
            .into_iter()
            .filter_map(|(style, text)| {
                let text = text.trim_end_matches(['\n', '\r']);
                (!text.is_empty())
                    .then(|| Span::styled(text.to_string(), style_from_syntect(style)))
            })
            .collect::<Vec<_>>();
        lines.push(Line::from(spans));
    }
    Some(lines)
}

pub(super) fn highlight_code(code: &str, language: &str) -> Vec<Line<'static>> {
    highlighted_lines(code, language).unwrap_or_else(|| {
        let mut lines = code
            .lines()
            .map(|line| Line::from(line.to_string()))
            .collect::<Vec<_>>();
        if lines.is_empty() {
            lines.push(Line::default());
        }
        lines
    })
}
