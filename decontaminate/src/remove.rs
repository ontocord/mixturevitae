// src/bin/filter_contaminated.rs
use std::{collections::HashSet, fs::File, io::{BufRead, BufReader, Write}, path::Path};

use arrow_array::StringArray;
use parquet::arrow::ParquetRecordBatchReader;
use serde::Serialize;

// Bring helpers from your crate:
// - get_word_re()
// - normalize_text_rust(&str, RegexLike)  [only used if you need it; not required to compute sub_doc_id]
// - doc_identity_rust(None, &str)
// - make_sub_doc_id(...)  <-- the tiny helper we added
use fast_decont::{get_word_re, doc_identity_rust}; // adjust the path
// If make_sub_doc_id is in same module as scan_file_rust, expose it with pub(crate) or pub
use fast_decont::make_sub_doc_id; // ensure it's exported

#[derive(Serialize)]
struct OutRec<'a> {
    doc_id: &'a str,
    text:   &'a str,
}

fn load_contaminated_ids<P: AsRef<Path>>(p: P) -> anyhow::Result<HashSet<String>> {
    let f = File::open(p)?;
    let r = BufReader::new(f);
    let mut set = HashSet::new();
    for line in r.lines() {
        let line = line?;
        let s = line.trim();
        if s.is_empty() { continue; }
        // Accept either raw ID per line or JSON with "doc_id"
        if s.starts_with('{') {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
                if let Some(id) = v.get("doc_id").and_then(|x| x.as_str()) {
                    set.insert(id.to_string());
                    continue;
                }
            }
        }
        set.insert(s.to_string());
    }
    Ok(set)
}

fn main() -> anyhow::Result<()> {
    // Minimal arg parsing:
    //   argv[1] = input parquet
    //   argv[2] = text_key
    //   argv[3] = id_key or "-" if none
    //   argv[4] = contaminated_ids file
    //   argv[5] = output jsonl path
    let mut args = std::env::args().skip(1);
    let in_parquet  = args.next().expect("in_parquet");
    let text_key    = args.next().expect("text_key");
    let id_key_arg  = args.next().expect("id_key_or_dash");
    let contam_path = args.next().expect("contaminated_ids");
    let out_path    = args.next().expect("out_jsonl");

    let id_key_opt = if id_key_arg == "-" { None } else { Some(id_key_arg.as_str()) };

    let contaminated = load_contaminated_ids(&contam_path)?;
    eprintln!("Loaded {} contaminated ids", contaminated.len());

    // Open parquet and iterate like scan_file_rust
    let file = File::open(Path::new(&in_parquet))?;
    let mut arrow_reader = ParquetRecordBatchReader::try_new(file, 1024)?;
    let schema = arrow_reader.schema();

    let text_col_idx = schema.index_of(&text_key)?;
    let id_col_idx = id_key_opt.and_then(|k| schema.index_of(k).ok());

    // Optional: same regex if you need tokenization; we don't need it just to compute sub_doc_id
    let _re = get_word_re();

    let mut out = std::io::BufWriter::new(File::create(&out_path)?);
    let mut kept = 0usize;
    let mut dropped = 0usize;

    while let Some(batch_result) = arrow_reader.next() {
        let batch = batch_result?;
        let text_array = batch.column(text_col_idx)
            .as_any().downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("Text column is not a string type"))?;
        let id_array = id_col_idx.map(|idx| {
            batch.column(idx).as_any().downcast_ref::<StringArray>()
        }).flatten();

        for i in 0..text_array.len() {
            if !text_array.is_valid(i) { continue; }
            let packed_text_original = text_array.value(i);
            if packed_text_original.is_empty() { continue; }
            // Mirror scan_file_rust normalization of the sentinel:
            let packed_text = packed_text_original.replace("<|endoftext}>", "<|endoftext|>");

            // Obtain original_id string if present and valid
            let original_id_val = id_array.and_then(|arr| if arr.is_valid(i) { Some(arr.value(i)) } else { None });

            for (sub_doc_index, sub_text) in packed_text.split("<|endoftext|>").enumerate() {
                if sub_text.is_empty() { continue; }

                // Build sub_doc_id exactly like scan_file_rust:
                let sub_doc_id = make_sub_doc_id(original_id_val, sub_doc_index, sub_text);

                // Filter
                if contaminated.contains(&sub_doc_id) {
                    dropped += 1;
                    continue;
                }

                // Write kept line as JSONL with doc_id + text
                // (Adjust schema to your needs; this is lightweight & line-based.)
                let rec = serde_json::json!({
                    "doc_id": sub_doc_id,
                    "text":   sub_text,
                });
                serde_json::to_writer(&mut out, &rec)?;
                out.write_all(b"\n")?;
                kept += 1;
            }
        }
    }

    out.flush()?;
    eprintln!("Kept: {}, Dropped: {}", kept, dropped);
    Ok(())
}
