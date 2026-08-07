use rust_latex_parser::{AccentKind, EqNode, parse_equation};

pub(crate) fn render_inline(source: &str) -> Option<String> {
    let source = source.trim();
    if source.is_empty() || !has_balanced_groups(source) {
        return None;
    }
    let rendered = render_linear(&parse_equation(source));
    valid_rendered(&normalize_operator_spacing(&rendered))
}

pub(crate) fn render_display(source: &str) -> Option<String> {
    let source = source.trim();
    if source.is_empty() || !has_balanced_groups(source) {
        return None;
    }
    valid_rendered(&term_maths::render(source).to_string())
}

fn valid_rendered(rendered: &str) -> Option<String> {
    let rendered = rendered
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n");
    (!rendered.trim().is_empty() && !rendered.contains('\\')).then_some(rendered)
}

fn has_balanced_groups(source: &str) -> bool {
    let mut depth = 0usize;
    let mut chars = source.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            let _ = chars.next();
            continue;
        }
        match ch {
            '{' => depth = depth.saturating_add(1),
            '}' => {
                let Some(next) = depth.checked_sub(1) else {
                    return false;
                };
                depth = next;
            }
            _ => {}
        }
    }
    depth == 0
}

fn render_linear(node: &EqNode) -> String {
    match node {
        EqNode::Text(text) | EqNode::TextBlock(text) => text.clone(),
        EqNode::Space(width) => if *width > 0.0 { " " } else { "" }.to_string(),
        EqNode::Seq(nodes) => {
            normalize_spaces(&nodes.iter().map(render_linear).collect::<String>())
        }
        EqNode::Sup(base, superscript) => {
            let base = render_linear(base);
            let script = render_linear(superscript);
            format!("{base}{}", render_script(&script, true))
        }
        EqNode::Sub(base, subscript) => {
            let base = render_linear(base);
            let script = render_linear(subscript);
            format!("{base}{}", render_script(&script, false))
        }
        EqNode::SupSub(base, superscript, subscript) => {
            let base = render_linear(base);
            let superscript = render_script(&render_linear(superscript), true);
            let subscript = render_script(&render_linear(subscript), false);
            format!("{base}{subscript}{superscript}")
        }
        EqNode::Frac(numerator, denominator) => {
            let numerator = render_linear(numerator);
            let denominator = render_linear(denominator);
            format!(
                "{}/{}",
                parenthesize_fraction_part(&numerator),
                parenthesize_fraction_part(&denominator)
            )
        }
        EqNode::Sqrt(content) => {
            let content = render_linear(content);
            if is_simple_math(&content) {
                format!("√{content}")
            } else {
                format!("√({content})")
            }
        }
        EqNode::BigOp {
            symbol,
            lower,
            upper,
        } => {
            let mut output = symbol.clone();
            if let Some(lower) = lower {
                output.push_str(&render_script(&render_linear(lower), false));
            }
            if let Some(upper) = upper {
                output.push_str(&render_script(&render_linear(upper), true));
            }
            output
        }
        EqNode::Accent(content, kind) => {
            let mut content = render_linear(content);
            let mark = match kind {
                AccentKind::Hat => '\u{0302}',
                AccentKind::Bar => '\u{0305}',
                AccentKind::Dot => '\u{0307}',
                AccentKind::DoubleDot => '\u{0308}',
                AccentKind::Tilde => '\u{0303}',
                AccentKind::Vec => '\u{20d7}',
            };
            content.push(mark);
            content
        }
        EqNode::Limit { name, lower } => lower.as_ref().map_or_else(
            || name.clone(),
            |lower| format!("{name}[{}]", render_linear(lower)),
        ),
        EqNode::MathFont { kind, content } => {
            term_maths::mathfont::map_str(kind, &render_linear(content))
        }
        EqNode::Delimited {
            left,
            right,
            content,
        } => format!("{left}{}{right}", render_linear(content)),
        EqNode::Matrix { rows, .. } => {
            let rows = rows
                .iter()
                .map(|row| row.iter().map(render_linear).collect::<Vec<_>>().join(", "))
                .collect::<Vec<_>>()
                .join("; ");
            format!("[{rows}]")
        }
        EqNode::Cases { rows } => rows
            .iter()
            .map(|(value, condition)| {
                let value = render_linear(value);
                condition.as_ref().map_or(value.clone(), |condition| {
                    format!("{value} if {}", render_linear(condition))
                })
            })
            .collect::<Vec<_>>()
            .join("; "),
        EqNode::Binom(top, bottom) => {
            format!("({} choose {})", render_linear(top), render_linear(bottom))
        }
        EqNode::Brace {
            content,
            label,
            over,
        } => {
            let content = render_linear(content);
            label.as_ref().map_or(content.clone(), |label| {
                let label = render_script(&render_linear(label), *over);
                format!("{content}{label}")
            })
        }
        EqNode::StackRel {
            base,
            annotation,
            over,
        } => format!(
            "{}{}",
            render_linear(base),
            render_script(&render_linear(annotation), *over)
        ),
    }
}

fn parenthesize_fraction_part(part: &str) -> String {
    if is_atomic_fraction_part(part) {
        part.to_string()
    } else {
        format!("({part})")
    }
}

fn is_atomic_fraction_part(text: &str) -> bool {
    if text.chars().all(|ch| ch.is_ascii_digit() || ch == '.') {
        return !text.is_empty();
    }
    let mut chars = text.chars();
    chars.next().is_some() && chars.all(is_script_glyph)
}

fn is_simple_math(text: &str) -> bool {
    !text.is_empty()
        && text
            .chars()
            .all(|ch| ch.is_alphanumeric() || is_math_alphanumeric(ch))
}

fn is_math_alphanumeric(ch: char) -> bool {
    matches!(ch, 'α'..='ω' | 'Α'..='Ω') || is_script_glyph(ch)
}

fn is_script_glyph(ch: char) -> bool {
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ".contains(ch)
}

fn render_script(text: &str, superscript: bool) -> String {
    text.chars()
        .map(|ch| script_char(ch, superscript))
        .collect::<Option<String>>()
        .unwrap_or_else(|| {
            if superscript {
                format!("^({text})")
            } else {
                format!("_({text})")
            }
        })
}

fn script_char(ch: char, superscript: bool) -> Option<char> {
    if superscript {
        Some(match ch {
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            '9' => '⁹',
            '+' => '⁺',
            '-' => '⁻',
            '=' => '⁼',
            '(' => '⁽',
            ')' => '⁾',
            'a' => 'ᵃ',
            'b' => 'ᵇ',
            'c' => 'ᶜ',
            'd' => 'ᵈ',
            'e' => 'ᵉ',
            'f' => 'ᶠ',
            'g' => 'ᵍ',
            'h' => 'ʰ',
            'i' => 'ⁱ',
            'j' => 'ʲ',
            'k' => 'ᵏ',
            'l' => 'ˡ',
            'm' => 'ᵐ',
            'n' => 'ⁿ',
            'o' => 'ᵒ',
            'p' => 'ᵖ',
            'r' => 'ʳ',
            's' => 'ˢ',
            't' => 'ᵗ',
            'u' => 'ᵘ',
            'v' => 'ᵛ',
            'w' => 'ʷ',
            'x' => 'ˣ',
            'y' => 'ʸ',
            'z' => 'ᶻ',
            _ => return None,
        })
    } else {
        Some(match ch {
            '0' => '₀',
            '1' => '₁',
            '2' => '₂',
            '3' => '₃',
            '4' => '₄',
            '5' => '₅',
            '6' => '₆',
            '7' => '₇',
            '8' => '₈',
            '9' => '₉',
            '+' => '₊',
            '-' => '₋',
            '=' => '₌',
            '(' => '₍',
            ')' => '₎',
            'a' => 'ₐ',
            'e' => 'ₑ',
            'h' => 'ₕ',
            'i' => 'ᵢ',
            'j' => 'ⱼ',
            'k' => 'ₖ',
            'l' => 'ₗ',
            'm' => 'ₘ',
            'n' => 'ₙ',
            'o' => 'ₒ',
            'p' => 'ₚ',
            'r' => 'ᵣ',
            's' => 'ₛ',
            't' => 'ₜ',
            'u' => 'ᵤ',
            'v' => 'ᵥ',
            'x' => 'ₓ',
            _ => return None,
        })
    }
}

fn normalize_spaces(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_operator_spacing(text: &str) -> String {
    const OPERATORS: [char; 24] = [
        '=', '≠', '<', '>', '≤', '≥', '≈', '≡', '∈', '∉', '⊂', '⊆', '⊃', '⊇', '→', '←', '↔', '⇒',
        '⇐', '⇔', '±', '∓', '×', '·',
    ];

    let mut output = String::with_capacity(text.len());
    for ch in text.chars() {
        if OPERATORS.contains(&ch) {
            while output.ends_with(' ') {
                output.pop();
            }
            if !output.is_empty() && !output.ends_with(['(', '[', '{']) {
                output.push(' ');
            }
            output.push(ch);
            output.push(' ');
        } else {
            output.push(ch);
        }
    }
    normalize_spaces(&output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_inline_and_display_math() {
        assert_eq!(
            render_inline(r"\mathbb{C}^3 \to \mathbb{C}^3"),
            Some("ℂ³ → ℂ³".to_string())
        );
        assert_eq!(
            render_inline(r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}"),
            Some("x = (-b ± √(b² - 4ac))/(2a)".to_string())
        );

        let display = render_display(r"\frac{x^2+1}{x-1}").unwrap();
        assert!(display.contains("x² + 1"));
        assert!(display.contains('─'));
        assert!(display.contains("x - 1"));
    }

    #[test]
    fn preserves_unsupported_or_incomplete_math() {
        assert_eq!(render_inline(r"x + \unknown{y}"), None);
        assert_eq!(render_display(r"\frac{a}{b"), None);
    }
}
