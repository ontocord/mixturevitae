# Decontamination

To run decontamination, please follow the following steps.
1. Install the Python dependencies from `requirements.txt`.
2. Build the Rust worker: `maturin develop --release`.
3. Build the index: `python decontam_hf.py build-index --out-index "$PICKLE_INDEX_PATH"`
4. Conver the index to native format: `python fast_decont/decontam_hf.py build-index-native  --in-pickle "$PICKLE_INDEX_PATH --out-native "$NATIVE_INDEX_PATH"`
5. Run the scan: ```python decontam_hf.py scan --input-glob "mixture_parquet/*.parquet"     --index "index_13gram.native"     --text-key "text"     --id-key "id"     --out-dir "/home/user/mixturevitae-paper/decontam_results/new"     --workers 64     --min-hits 3``
