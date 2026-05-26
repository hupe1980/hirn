use hirn::agent::AgentId;
use hirn::episodic::EpisodicRecord;
use hirn_engine::ExportData;

fn main() {
    let agent = AgentId::new("test_agent").unwrap();
    let mut records = Vec::new();
    for i in 0..5usize {
        let record = EpisodicRecord::builder()
            .content(&format!("event {i}"))
            .agent_id(agent.clone())
            .embedding(vec![0.1 + (i as f32 * 0.01); 768])
            .build()
            .unwrap();
        records.push(record);
    }

    let export_data = ExportData {
        version: 1,
        working: vec![],
        episodic: records,
        semantic: vec![],
        procedural: vec![],
        agents: vec![],
        namespaces: vec![],
        edges: vec![],
    };

    let json = serde_json::to_string(&export_data).unwrap();
    print!("{json}");
}
