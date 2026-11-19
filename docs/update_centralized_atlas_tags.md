# Managing Centralized Atlas Tags

The shared Google Sheet can be updated by InChIKey:

```bash
metatlas2 update-tags retrieve
metatlas2 update-tags add --input-csv tags.csv
metatlas2 update-tags remove --input-csv tags.csv
metatlas2 update-tags retag --input-csv tags.csv
```

The CSV must have exactly the headers `inchi_key,tags`, in that order:

```csv
inchi_key,tags
GTZCVFVGUGFEME-IWQZZHSRSA-N,EMA;PlantHormone;I like this compound
GFFGJBXGBJISGV-UHFFFAOYSA-N,NewSet
```

Keys must use the uppercase canonical 14-10-1 format and may occur only once
in the CSV. Tags must be nonempty semicolon-separated tokens; leading and
trailing whitespace is ignored, and matching is case-sensitive. Each operation
applies to **every sheet row** with that key. `add` appends tags without
duplicates, `remove` removes exact tokens (and can clear the cell), and `retag`
replaces all tags. Missing keys, existing additions, and missing removals emit
warnings. `retrieve` prints sorted unique tags, one per line.

The entire CSV is validated before any write. Only changed tag cells are
written, using literal values. The command checks for sheet changes before
writing, but this is not an atomic lock: coordinate with other editors, since
edits or row sorting between that check and the write can still cause conflicts.
The existing atlas service account must have access to the sheet; on Docker,
its standard credential file must also be available on the host. Existing
project atlas snapshots are not changed by these edits.
