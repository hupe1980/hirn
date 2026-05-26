#[cfg(feature = "cedar")]
mod tests {
    use cedar_policy::{PolicySet, Schema, ValidationMode, Validator};
    use hirn_policy::DEFAULT_SCHEMA;

    const CEDAR_GUIDE_DOC: &str = include_str!("../../../docs/cedar-guide.md");
    const CEDAR_PATTERNS_DOC: &str = include_str!("../../../docs/cedar-patterns.md");

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
    fn cedar_patterns_validate_against_live_schema() {
        let schema: Schema = DEFAULT_SCHEMA.parse().expect("default schema should parse");
        let validator = Validator::new(schema);
        for (doc_name, markdown) in [
            ("cedar-patterns", CEDAR_PATTERNS_DOC),
            ("cedar-guide", CEDAR_GUIDE_DOC),
        ] {
            let snippets = fenced_blocks(markdown, "cedar");
            assert!(
                !snippets.is_empty(),
                "expected Cedar examples in {doc_name}"
            );

            for (index, snippet) in snippets.iter().enumerate() {
                let policies: PolicySet = snippet.parse().unwrap_or_else(|error| {
                    panic!("{doc_name} snippet {index} should parse: {error}");
                });
                let result = validator.validate(&policies, ValidationMode::default());
                assert!(
                    result.validation_passed(),
                    "{doc_name} snippet {index} should validate against the live schema: {result:?}"
                );
            }
        }
    }
}
