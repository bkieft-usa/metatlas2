"""Raw-data ingest: .raw -> .mzML -> .h5 conversion and the LIMS file-table sync.

Ported from the legacy ``metatlas-repo/utils/{convert_files.sh, raw_to_h5.sh}``,
``metatlas/io/{file_converter.py, mzml_loader.py}`` and
``metatlas/io/update_lcmsfiles_in_lims.py``. See ``docs/raw_data_ingest.md``.

Modules that run on the *host* (outside the container, standard library only,
so they work with ``/usr/bin/python3.11`` for any account):

conversion           Scan for unconverted .raw files and run the per-file pipeline.
run_file_conversion  CLI for the converter.

Modules that run *inside* the metatlas2 container:

h5_writer            mzML -> metatlas HDF5 (format version 5).
lims_file_sync       Sync raw/mzML/h5 files on disk into the LIMS file tables.
run_lims_file_sync   CLI for the LIMS sync.
"""
