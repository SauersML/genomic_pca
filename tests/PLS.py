import numpy as np
from pathlib import Path
from tqdm import tqdm
from bed_reader import open_bed
from pgenlib import PgenWriter

IN  = Path("hg38_plink1")
OUT = Path("hg38_chr22")

# ---------- helpers ----------
def is_chr22(c: str) -> bool:
    c = c.strip()
    if c.lower().startswith("chr"): c = c[3:]
    return c == "22"

# ---------- write .psam from .fam (and count samples) ----------
fam_path = IN.with_suffix(".fam")
n_samples = 0
with open(OUT.with_suffix(".psam"), "w") as g, open(fam_path) as f:
    g.write("#FID\tIID\tPAT\tMAT\tSEX\tPHENOTYPE\n")
    for ln in f:
        g.write(ln.replace(" ", "\t"))
        n_samples += 1

# ---------- pre-open .bed to know total variants (for % on BIM scan) ----------
bed = open_bed(str(IN.with_suffix(".bed")), count_A1=True)
total_variants = bed.sid_count

# ---------- scan .bim → collect chr22 indexes, write .pvar (with % progress) ----------
keep_idx = []
bim_path = IN.with_suffix(".bim")
with open(bim_path) as f, open(OUT.with_suffix(".pvar"), "w") as g, \
     tqdm(total=total_variants, unit="var", desc="Scanning .bim (chr22)", dynamic_ncols=True) as pbar:
    g.write("##fileformat=pvar\n#CHROM\tPOS\tID\tREF\tALT\n")
    for i, ln in enumerate(f):
        # BIM: chrom sid cm pos a1 a2
        c, sid, _, pos, a1, a2 = ln.rstrip().split()[:6]
        if is_chr22(c):
            g.write(f"{c}\t{pos}\t{sid}\t{a2}\t{a1}\n")  # REF=a2, ALT=a1 (A1 counts → ALT)
            keep_idx.append(i)
        pbar.update(1)

keep = np.asarray(keep_idx, dtype=np.uint32)
m_variants = int(keep.size)
if m_variants == 0:
    bed.close()
    raise SystemExit("No chr22 variants found in BIM.")

# ---------- create .pgen writer (filename must be BYTES for pgenlib) ----------
pgen_out_bytes = str(OUT.with_suffix(".pgen")).encode()
pw = PgenWriter(pgen_out_bytes, n_samples, m_variants, True)  # nonref_flags=True for BED-origin

# ---------- stream BED → write PGEN with % and units ----------
# We iterate the BED variant axis in blocks, pick the chr22 columns in each block, and append.
blk = 1 << 13  # 8192 variants per read; adjust if you want smaller memory spikes
written = 0
with tqdm(total=m_variants, unit="var", desc="Writing chr22 → .pgen", dynamic_ncols=True) as pbar:
    for start in range(0, total_variants, blk):
        end = min(start + blk, total_variants)
        # local indices within this window
        sel = keep[(keep >= start) & (keep < end)]
        if sel.size == 0:
            continue
        sel_local = sel - start

        # X: (samples, window_width) with values {0,1,2,nan} for A1 counts
        X = bed.read(index=np.s_[:, start:end], dtype="float32", order="C")
        X = X[:, sel_local]                                # (samples, kept_in_block)
        miss = np.isnan(X)                                 # mask NaNs before cast
        X = X.astype(np.int8, copy=False)                  # cast to int8
        X[miss] = -9                                       # PLINK missing code
        slab = np.ascontiguousarray(X.T)                   # (variants, samples), C-contiguous
        pw.append_biallelic_batch(slab)

        written += slab.shape[0]
        pbar.update(slab.shape[0])

# ---------- close handles and final check ----------
pw.close()
bed.close()
assert written == m_variants, f"Variant count mismatch: wrote {written}, expected {m_variants}"

print(f"Done.\nWrote:\n  {OUT.with_suffix('.pgen')}\n  {OUT.with_suffix('.pvar')}\n  {OUT.with_suffix('.psam')}")
