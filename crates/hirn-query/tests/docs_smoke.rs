use hirn_query::{AnalyzeContext, analyze, parse};

const TROUBLESHOOTING_DOC: &str = include_str!("../../../docs/troubleshooting.md");

fn fenced_blocks(markdown: &str, language: &str) -> Vec<String> {
    let fence = format!("```{language}");
    let mut blocks = Vec::new();
    let mut current: Option<String> = None;

    for line in markdown.lines() {
        let trimmed = line.trim();
        if current.is_none() && trimmed == fence {
            current = Some(String::new());
            continue;
        }

        if let Some(block) = &mut current {
            if trimmed == "```" {
                blocks.push(std::mem::take(block));
                current = None;
            } else {
                block.push_str(line);
                block.push('\n');
            }
        }
    }

    blocks
}

#[test]
fn troubleshooting_hirnql_examples_parse_and_analyze() {
    let snippets = fenced_blocks(TROUBLESHOOTING_DOC, "sql");
    let ctx = AnalyzeContext::default();

    assert_eq!(
        snippets.len(),
        1,
        "expected one HirnQL example block in troubleshooting doc"
    );

    let statements = snippets[0]
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("--"))
        .map(|line| line.trim_end_matches(';'));

    for statement in statements {
        let ast = parse(statement)
            .unwrap_or_else(|error| panic!("statement should parse: {statement}: {error}"));
        analyze(&ast, &ctx).unwrap_or_else(|error| {
            panic!("statement should analyze: {statement}: {error}");
        });
    }
}
