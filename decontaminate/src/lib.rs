use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet};
use pyo3::exceptions::PyValueError;
use std::io::{BufReader, BufWriter, Write};

use hashbrown::{HashSet, HashMap};
use once_cell::sync::OnceCell;
use unicode_normalization::UnicodeNormalization;
use regex::Regex;
use blake2::{Blake2b, Digest};
use blake2::digest::consts::{U8, U16};
use std::fs::File;

use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian};
use std::cell::RefCell;
use std::path::Path;

use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use arrow::array::{Array, StringArray};
use arrow::record_batch::RecordBatchReader;

use serde::Serialize;
use serde_json;

static HASH_INDEX: OnceCell<HashSet<u64>> = OnceCell::new();
static ATTR_INDEX: OnceCell<HashMap<u64, u64>> = OnceCell::new();
static WORD_RE: OnceCell<Regex> = OnceCell::new();

static STOP_WORDS: OnceCell<HashSet<&'static str>> = OnceCell::new();
fn get_stop_words() -> &'static HashSet<&'static str> {
    STOP_WORDS.get_or_init(|| {
        [
            "a","an","and","are","as","at","be","by","for","from","has","he",
            "in","is","it","its","of","on","that","the","to","was","were",
            "will","with","what","which","who","when","where","why","how",
            "this","that","these","those","or","if","then","else","not",
        ].iter().cloned().collect()
    })
}

thread_local! {
    static NORM_BUF: RefCell<String> = RefCell::new(String::with_capacity(1024));
    static SEEN_LOCAL: RefCell<HashSet<u64>> = RefCell::new(HashSet::new());
}

fn get_word_re() -> &'static Regex {
    WORD_RE.get_or_init(|| Regex::new(r"[a-z0-9]+(?:'[a-z0-9]+)?").unwrap())
}

fn load_index_from_native(path: &str) -> PyResult<HashSet<u64>> {
    let file = File::open(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to open index file: {}", e)))?;
    let mut reader = BufReader::new(file);
    let mut hashes = HashSet::new();
    loop {
        match reader.read_u64::<LittleEndian>() {
            Ok(hash) => { hashes.insert(hash); },
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(PyValueError::new_err(format!("Error reading index file: {}", e))),
        }
    }
    Ok(hashes)
}

fn load_attr_from_sidecar(index_path: &str) -> PyResult<HashMap<u64, u64>> {
    // auto-discover sidecar: "<index>.attr"
    let attr_path = format!("{}.attr", index_path);
    let file = File::open(&attr_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to open attribution sidecar '{}': {}", attr_path, e)))?;
    let mut reader = BufReader::new(file);
    let mut map = HashMap::new();
    loop {
        let h = match reader.read_u64::<LittleEndian>() {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(PyValueError::new_err(format!("Error reading attr hash: {}", e))),
        };
        let mask = reader.read_u64::<LittleEndian>()
            .map_err(|e| PyValueError::new_err(format!("Error reading attr mask: {}", e)))?;
        map.insert(h, mask);
    }
    Ok(map)
}

fn get_hash_index(path: &str) -> PyResult<&'static HashSet<u64>> {
    HASH_INDEX.get_or_try_init(|| load_index_from_native(path))
}
fn get_attr_index(path: &str) -> PyResult<&'static HashMap<u64, u64>> {
    ATTR_INDEX.get_or_try_init(|| load_attr_from_sidecar(path))
}

fn normalize_text_rust(s: &str, re: &'static Regex) -> Vec<String> {
    if s.is_empty() { return Vec::new(); }
    let stop_words = get_stop_words();
    NORM_BUF.with_borrow_mut(|norm_buf| {
        norm_buf.clear();
        s.nfkc().for_each(|c| norm_buf.push(c));
        let lower_s = norm_buf.to_lowercase();
        re.find_iter(&lower_s)
          .map(|m| m.as_str())
          .filter(|s| { !stop_words.contains(*s) })
          .map(|s| s.to_string())
          .collect()
    })
}

fn hash_ngram_window(win: &[String]) -> u64 {
    let mut hasher = Blake2b::<U8>::new();
    for (j, word) in win.iter().enumerate() {
        if j > 0 { hasher.update(b" "); }
        hasher.update(word.as_bytes());
    }
    let res: [u8; 8] = hasher.finalize().into();
    u64::from_le_bytes(res)
}

fn doc_identity_rust(id_val: Option<&str>, text: &str) -> String {
    if let Some(id_str) = id_val {
        if !id_str.is_empty() { return id_str.to_string(); }
    }
    let sample = if text.len() <= 1024 {
        text
    } else {
        let mut end_idx = 1024;
        while !text.is_char_boundary(end_idx) { end_idx -= 1; }
        &text[..end_idx]
    };
    let mut hasher = Blake2b::<U16>::new();
    hasher.update(sample.as_bytes());
    let res: [u8; 16] = hasher.finalize().into();
    hex::encode(res)
}

#[derive(Serialize)]
struct FileScanResult {
    doc_id: String,
    match_count: usize,
    // (source_id, hits) for all sources with >=1 hit
    src_hits: Vec<(u32, usize)>,
}

#[pyfunction]
fn build_index_native_rust(py: Python, in_pickle_path: &str, out_native_path: &str) -> PyResult<()> {
    let pickle = py.import("pickle")?;
    let f = py.import("builtins")?.call_method1("open", (in_pickle_path, "rb"))?;
    let data: &PyDict = pickle.call_method1("load", (f,))?.extract()?;

    // ---- MAIN INDEX ----
    // 1) PyResult<Option<&PyAny>>  ->  ? to get Option<&PyAny>
    // 2) Option<&PyAny> -> ok_or_else to get &PyAny
    let hashes_py_opt = data.get_item("hashes")?;
    let hashes_py = hashes_py_opt
        .ok_or_else(|| PyValueError::new_err("Pickle missing 'hashes'"))?;
    let hashes: &PySet = hashes_py
        .downcast()
        .map_err(|_| PyValueError::new_err("'hashes' must be a set"))?;

    let mut idx_file = BufWriter::new(
        File::create(out_native_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to create native index: {}", e)))?
    );
    for hash_py in hashes {
        let h: u64 = hash_py.extract()?;
        idx_file.write_u64::<LittleEndian>(h)?;
    }
    idx_file.flush()
        .map_err(|e| PyValueError::new_err(format!("Failed to flush native index: {}", e)))?;

    // ---- OPTIONAL ATTRIBUTION SIDECAR ----
    // get_item returns PyResult<Option<&PyAny>>
    if let Ok(Some(src_any)) = data.get_item("sources") {
        if let Ok(src_dict) = src_any.downcast::<PyDict>() {
            use hashbrown::HashMap;
            let mut pairs_mask: Vec<(u64, u64)> = Vec::with_capacity(src_dict.len());
            let mut legend: Vec<String> = Vec::new();
            let mut name_to_id: HashMap<String, u32> = HashMap::new();

            // detect value type
            let mut is_mask = None;
            for (_k, v) in src_dict.iter() {
                if v.downcast::<pyo3::types::PyLong>().is_ok() { is_mask = Some(true); }
                else if v.downcast::<pyo3::types::PyString>().is_ok() { is_mask = Some(false); }
                break;
            }
            let is_mask = is_mask.unwrap_or(true);

            if is_mask {
                for (k, v) in src_dict.iter() {
                    let h: u64 = k.extract()?;
                    let m: u64 = v.extract()?;
                    pairs_mask.push((h, m));
                }
            } else {
                let mut tmp_masks: HashMap<u64, u64> = HashMap::new();
                for (k, v) in src_dict.iter() {
                    let h: u64 = k.extract()?;
                    let name: String = v.extract()?;
                    let id = *name_to_id.entry(name.clone()).or_insert_with(|| {
                        let nid = legend.len() as u32;
                        legend.push(name);
                        nid
                    });
                    let bit: u64 = 1u64 << (id as u64);
                    *tmp_masks.entry(h).or_insert(0) |= bit;
                }
                pairs_mask.extend(tmp_masks.into_iter());
            }

            let attr_path = format!("{}.attr", out_native_path);
            let mut w = BufWriter::new(
                File::create(&attr_path)
                    .map_err(|e| PyValueError::new_err(format!("Failed to create '{}': {}", attr_path, e)))?
            );
            for (h, m) in pairs_mask {
                w.write_u64::<LittleEndian>(h)?;
                w.write_u64::<LittleEndian>(m)?;
            }
            w.flush().map_err(|e| PyValueError::new_err(format!("Failed to flush '{}': {}", attr_path, e)))?;

            if !legend.is_empty() {
                let json_path = format!("{}.sources.json", out_native_path);
                std::fs::write(&json_path, serde_json::to_vec_pretty(&legend).unwrap())
                    .map_err(|e| PyValueError::new_err(format!("write '{}': {}", json_path, e)))?;
            }
        }
    }

    let print = py.import("builtins")?.getattr("print")?;
    print.call1((format!(
        "[build-native-rust] Wrote {} hashes to {}\n[build-native-rust] Attribution sidecar: {}",
        hashes.len(),
        out_native_path,
        // `contains` returns PyResult<bool> in newer PyO3
        if data.contains("sources")? { "yes" } else { "no" }
    ),))?;
    Ok(())
}





#[pyfunction]
#[pyo3(signature = (file_path, text_key, id_key, index_path, n, min_hits, min_coverage, out_hits_path=None))]
fn scan_file_rust(
    py: Python,
    file_path: &str,
    text_key: &str,
    id_key: Option<&str>,
    index_path: &str,
    n: usize,
    min_hits: usize,
    min_coverage: f64,
    out_hits_path: Option<&str>,
) -> PyResult<(usize, usize, Vec<String>)> {
    if min_hits == 0 { return Err(PyValueError::new_err("min_hits must be >= 1")); }
    if !(0.0..=1.0).contains(&min_coverage) {
        return Err(PyValueError::new_err("min_coverage must be in [0,1]"));
    }

    let index_hashes = get_hash_index(index_path)?;
    let attr_index  = get_attr_index(index_path)?;
    let re = get_word_re();

    let text_key_owned = text_key.to_string();
    let id_key_owned = id_key.map(|s| s.to_string());

    let file_processing_result = py.allow_threads(move || {
        // This HashMap is local to this closure
        let mut matched_hash_to_mask: HashMap<u64, u64> = HashMap::new();
        
        let file = File::open(Path::new(file_path))
            .map_err(|e| format!("Failed to open parquet file: {}", e))?;
        let mut arrow_reader = ParquetRecordBatchReader::try_new(file, 1024)
            .map_err(|e| e.to_string())?;
        let schema = arrow_reader.schema();

        let text_col_idx = schema.index_of(&text_key_owned)
            .map_err(|e| format!("Text key '{}' not found in schema: {}", text_key_owned, e.to_string()))?;
        let id_col_idx = id_key_owned.as_ref().and_then(|key| schema.index_of(key).ok());

        let mut total_scanned = 0usize;
        let mut total_contaminated = 0usize;
        let mut results_vec: Vec<String> = Vec::new();

        while let Some(batch_result) = arrow_reader.next() {
            let batch = batch_result.map_err(|e| e.to_string())?;
            let text_array = batch.column(text_col_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| "Text column is not a string type".to_string())?;
            let id_array = id_col_idx.map(|idx| {
                batch.column(idx).as_any().downcast_ref::<StringArray>()
            }).flatten();

            for i in 0..text_array.len() {
                if !text_array.is_valid(i) { continue; }
                let packed_text_original = text_array.value(i);
                if packed_text_original.is_empty() { continue; }
                let packed_text = packed_text_original.replace("<|endoftext}>", "<|endoftext|>");
                let original_id_val = id_array.and_then(|arr| if arr.is_valid(i) { Some(arr.value(i)) } else { None });

                for (sub_doc_index, sub_text) in packed_text.split("<|endoftext|>").enumerate() {
                    if sub_text.is_empty() || (sub_text.len() < n * 2) { continue; }
                    total_scanned += 1;

                    let toks = normalize_text_rust(sub_text, re);
                    if toks.len() < n { continue; }

                    // unique 13-gram shingles
                    let mut doc_shingles: HashSet<u64> = HashSet::new();
                    for win in toks.windows(n) {
                        doc_shingles.insert(hash_ngram_window(win));
                    }
                    let denom = doc_shingles.len().max(1) as f64;

                    // count distinct hits and by-source hits
                    let mut distinct_hits = 0usize;
                    let mut hits_by_src: HashMap<u32, usize> = HashMap::new();

                    for h in doc_shingles.iter() {
                        if index_hashes.contains(h) {
                            distinct_hits += 1;

                            let mask = attr_index.get(h).copied().unwrap_or(0);
                            // record by-source counts
                            if mask != 0 {
                                let mut m = mask;
                                while m != 0 {
                                    let bit = m.trailing_zeros();
                                    *hits_by_src.entry(bit).or_insert(0) += 1;
                                    m &= m - 1;
                                }
                            }
                            // record this hash for the per-file union
                            let entry = matched_hash_to_mask.entry(*h).or_insert(0);
                            *entry |= mask;
                        }
                    }

                    let coverage = (distinct_hits as f64) / denom;

                    if distinct_hits >= min_hits && coverage >= min_coverage {
                        total_contaminated += 1;

                        let sub_doc_id = if let Some(original_id) = original_id_val {
                            if !original_id.is_empty() {
                                format!("{}-part-{}", original_id, sub_doc_index)
                            } else {
                                doc_identity_rust(None, sub_text)
                            }
                        } else {
                            doc_identity_rust(None, sub_text)
                        };

                        let mut src_hits_vec: Vec<(u32, usize)> = Vec::with_capacity(hits_by_src.len());
                        for (k,v) in hits_by_src.into_iter() {
                            if v > 0 { src_hits_vec.push((k, v)); }
                        }

                        let rec = FileScanResult {
                            doc_id: sub_doc_id,
                            match_count: distinct_hits,
                            src_hits: src_hits_vec,
                        };
                        results_vec.push(serde_json::to_string(&rec).map_err(|e| e.to_string())?);
                    }
                }
            }
        } // end while

        if let Some(path) = out_hits_path {
            let file = File::create(path)
                .map_err(|e| format!("Failed to create per-hits file '{}': {}", path, e))?;
            let mut w = BufWriter::new(file);
            for (h, m) in matched_hash_to_mask.into_iter() {
                w.write_u64::<LittleEndian>(h).map_err(|e| e.to_string())?;
                w.write_u64::<LittleEndian>(m).map_err(|e| e.to_string())?;
            }
            w.flush().map_err(|e| e.to_string())?; 
        }
   
        Ok::<_, String>((total_scanned, total_contaminated, results_vec))
    }); // end py.allow_threads


    match file_processing_result {
        Ok(tuple) => Ok(tuple),
        Err(e_str) => Err(PyValueError::new_err(format!("Rust worker failed for {}: {}", file_path, e_str))),
    }
}

#[pymodule]
fn fast_decont(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_index_native_rust, m)?)?;
    m.add_function(wrap_pyfunction!(scan_file_rust, m)?)?;
    Ok(())
}
