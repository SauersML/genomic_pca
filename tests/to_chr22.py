import numpy as np
from pathlib import Path
from tqdm import tqdm
from bed_reader import open_bed
from pgenlib import PgenWriter

IN  = Path("hg38_plink1")
OUT = Path("hg38_chr22")

def is_chr22(c: str) -> bool:
    c = c.strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    return c == "22"

# ---------- .psam from .fam (count samples) ----------
fam_path = IN.with_suffix(".fam")
n_samples = 0
with open(OUT.with_suffix(".psam"), "w") as g, open(fam_path) as f:
    g.write("#FID\tIID\tPAT\tMAT\tSEX\tPHENOTYPE\n")
    for ln in f:
        g.write(ln.replace(" ", "\t"))
        n_samples += 1

# ---------- open BED first (for total-variant count) ----------
with open_bed(str(IN.with_suffix(".bed")), count_A1=True) as bed:
    total_variants = bed.sid_count

# ---------- scan .bim → chr22 indices; write .pvar (progress) ----------
keep_idxs = []
bim_path = IN.with_suffix(".bim")
with open(bim_path) as f_in, open(OUT.with_suffix(".pvar"), "w") as f_pvar, \
     tqdm(total=total_variants, unit="var", desc="Scanning .bim (chr22)", dynamic_ncols=True) as pbar:
    f_pvar.write("##fileformat=pvar\n#CHROM\tPOS\tID\tREF\tALT\n")
    for i, ln in enumerate(f_in):
        # BIM columns: chrom sid cm pos a1 a2
        c, sid, _, pos, a1, a2 = ln.rstrip().split()[:6]
        if is_chr22(c):
            # A1 is ALT in PLINK1 A1-count encoding; set REF=a2, ALT=a1
            f_pvar.write(f"{c}\t{pos}\t{sid}\t{a2}\t{a1}\n")
            keep_idxs.append(i)
        pbar.update(1)

keep = np.asarray(keep_idxs, dtype=np.uint32)
m_variants = int(keep.size)
if m_variants == 0:
    raise SystemExit("No chr22 variants found in BIM.")

# ---------- PGEN writer (filename must be BYTES for pgenlib) ----------
pgen_out = str(OUT.with_suffix(".pgen")).encode("utf-8")
pw = PgenWriter(pgen_out, n_samples, m_variants, True)  # nonref_flags=True for BED-origin genotypes

# ---------- stream BED → write PGEN (progress) ----------
blk = 1 << 13  # 8192 variants per window; adjust for memory
written = 0
with open_bed(str(IN.with_suffix(".bed")), count_A1=True) as bed, \
     tqdm(total=m_variants, unit="var", desc="Writing chr22 → .pgen", dynamic_ncols=True) as pbar:
    for start in range(0, bed.sid_count, blk):
        end = min(start + blk, bed.sid_count)

        # select chr22 indices that fall in this window
        sel_global = keep[(keep >= start) & (keep < end)]
        if sel_global.size == 0:
            continue
        sel_local = sel_global - start

        # X: (samples, window_width) with values {0,1,2,nan} for A1 counts
        X = bed.read(index=np.s_[:, start:end], dtype="float32", order="C")
        X = X[:, sel_local]  # keep only chr22 columns in this window

        # replace NaNs with sentinel BEFORE casting to int to avoid warnings
        miss = np.isnan(X)
        if miss.any():
            X = X.copy()
            X[miss] = -9.0

        # cast and arrange (variants, samples) contiguous for writer
        X = X.astype(np.int8, copy=False)          # now values are {0,1,2,-9}
        slab = np.ascontiguousarray(X.T)           # (variants, samples)

        pw.append_biallelic_batch(slab)
        wrote_now = slab.shape[0]
        written += wrote_now
        pbar.update(wrote_now)

pw.close()
assert written == m_variants, f"Variant count mismatch: wrote {written}, expected {m_variants}"

print("Done.")
print(f"  {OUT.with_suffix('.pgen')}")
print(f"  {OUT.with_suffix('.pvar')}")
print(f"  {OUT.with_suffix('.psam')}")
