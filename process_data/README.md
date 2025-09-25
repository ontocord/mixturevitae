# Data Curation

This folder contains the code for compiling the MixtureVitae dataset. Here are some components and the associated scripts. 

**Note**: The scripts have been tested and used in HPC environments using SLURM jobs.

The main script is `mixture_paper_data_curation.py`. This is the primary data curation pipeline that processes raw text data from multiple sources (Common-Pile, curated datasets, FineFine, Nemo, MAGA, txt360). It performs:

- Document and sentence-level deduplication
- Copyright and license filtering
- Text quality assessment using stopword and special character scoring
- Content cleanup (citation removal, HTML unescaping, junk line removal)
- URL standardization and metadata extraction
- Parallel processing across multiple nodes using SLURM

The script filters out non-permissive content, removes duplicates, and ensures compliance with copyright requirements while maintaining high-quality text for the final dataset.