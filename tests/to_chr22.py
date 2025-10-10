import numpy as np
from pathlib import Path
from bed_reader import open_bed
from pgenlib import PgenWriter

IN, OUT = Path("hg38_plink1"), Path("hg38_chr22")

# --- write .psam from .fam (header + tabbed rows); count samples
fam = IN.with_suffix(".fam")
n = 0
with open(OUT.with_suffix(".psam"), "w") as g, open(fam) as f:
    g.write("#FID\tIID\tPAT\tMAT\tSEX\tPHENOTYPE\n")
    for ln in f:
        g.write(ln.replace(" ", "\t")); n += 1

# --- stream .bim → build chr22 keep mask indices and write .pvar
def chrom22(c):
    c = c.strip()
    if c.lower().startswith("chr"): c = c[3:]
    return c == "22"
keep_idxs = []
with open(IN.with_suffix(".bim")) as f, open(OUT.with_suffix(".pvar"), "w") as g:
    g.write("##fileformat=pvar\n#CHROM\tPOS\tID\tREF\tALT\n")
    for i, ln in enumerate(f):
        c, sid, _, pos, a1, a2 = ln.rstrip().split()[:6]
        if chrom22(c):
            g.write(f"{c}\t{pos}\t{sid}\t{a2}\t{a1}\n")  # REF=a2, ALT=a1 (A1 counts → ALT)
            keep_idxs.append(i)
keep = np.asarray(keep_idxs, dtype=np.uint32)
m = int(keep.size)
if m == 0:
    raise SystemExit("No chr22 variants found.")

# --- read .bed in blocks (A1 counts), write .pgen (variant-major batches)
bed = open_bed(str(IN.with_suffix(".bed")), count_A1=True)
pw  = PgenWriter(str(OUT.with_suffix(".pgen")), n, m, True)  # nonref_flags=True for BED-origin
blk = 1 << 13  # 8192
written = 0
for start in range(0, bed.sid_count, blk):
    end = min(start + blk, bed.sid_count)
    sel = keep[(keep >= start) & (keep < end)] - start
    if sel.size == 0:
        continue
    X = bed.read(index=np.s_[:, start:end], dtype="float32", order="C")  # (n_samples, width)
    X = X[:, sel]  # keep only chr22 cols in this window
    miss = np.isnan(X)
    X = X.astype(np.int8, copy=False)
    X[miss] = -9
    pw.append_biallelic_batch(X.T)  # (variants, samples)
    written += X.shape[1]

assert written == m, f"wrote {written} variants, expected {m}"
pw.close(); bed.close()
print(f"wrote: {OUT.with_suffix('.pgen')}, {OUT.with_suffix('.pvar')}, {OUT.with_suffix('.psam')}")
