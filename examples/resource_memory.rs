//! # Resource Memory — first-class evidence, not text-only attachments.
//!
//! Run with: `cargo run --example resource_memory -p hirn`

use std::io::Cursor;

use hirn::prelude::*;
use hirn::resource::{DerivedArtifactKind, EvidenceRole, HydrationMode};
use image::{DynamicImage, ImageFormat, RgbaImage};
use tempfile::tempdir;

#[tokio::main]
async fn main() -> HirnResult<()> {
    let dir = tempdir().expect("failed to create temp dir");
    let path = dir.path().join("brain");
    let memory = HirnMemory::open(&path).await?;

    let agent = AgentId::new("resource_demo")?;
    memory
        .db()
        .register_agent(&agent, "Resource Memory Demo")
        .await?;

    let record = EpisodicRecord::builder()
        .content("Checkout failed in staging")
        .agent_id(agent)
        .multi_content(MemoryContent::Image {
            data: valid_png_bytes(),
            mime_type: "image/png".into(),
            description: "checkout page showing a card declined banner in staging".into(),
        })
        .build()?;
    let id = memory.db().episodic().remember(record).await?;
    println!("✓ Stored image-backed episode: {id}");

    let query = memory
        .db()
        .embed_text("card declined checkout screenshot")
        .await?;
    let recalled = memory
        .db()
        .recall_view()
        .query(query)
        .agent_id(agent.as_str())
        .limit(3)
        .execute()
        .await?;

    let result = recalled
        .iter()
        .find(|candidate| candidate.record.id() == id)
        .expect("stored image episode should be recalled");
    let source = result
        .resource_evidence
        .iter()
        .find(|summary| summary.role == EvidenceRole::Source && summary.artifact_kind.is_none())
        .expect("source resource evidence should be present");
    println!(
        "✓ Recalled resource {} with artifacts {:?}",
        source.resource_id, source.available_artifacts
    );

    assert!(
        source
            .available_artifacts
            .contains(&DerivedArtifactKind::Thumbnail)
    );
    let preview = memory
        .db()
        .recall_view()
        .fetch_resource(&agent, source.resource_id, HydrationMode::Preview)
        .await?
        .expect("preview hydration should find the stored resource");
    let thumbnail = preview
        .artifacts
        .iter()
        .find(|artifact| artifact.kind == DerivedArtifactKind::Thumbnail)
        .expect("preview hydration should expose the thumbnail artifact");

    println!(
        "✓ Preview hydration returned {} artifact(s); thumbnail MIME: {}",
        preview.artifacts.len(),
        thumbnail.mime_type.as_deref().unwrap_or("unknown")
    );

    Ok(())
}

fn valid_png_bytes() -> Vec<u8> {
    let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
        4,
        4,
        image::Rgba([0x22, 0x66, 0xaa, 0xff]),
    ));
    let mut bytes = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
        .expect("png fixture should encode");
    bytes
}
