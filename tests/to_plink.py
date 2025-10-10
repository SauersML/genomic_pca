from __future__ import annotations
import io
import sys
from pathlib import Path

import numpy as np
import zstandard as zstd

try:
    import pgenlib as pg  # PLINK2 reader (Python API)
except Exception:
    print("ERROR: Please 'pip install --user Pgenlib zstandard numpy' first.", file=sys.stderr)
    raise

# ---------- helpers ----------

def open_text(path: Path):
    """Open plain or .zst text as text stream."""
    if path.suffix == ".zst":
        dctx = zstd.ZstdDecompressor()
        return io.TextIOWrapper(dctx.stream_reader(open(path, "rb")), encoding="utf-8")
    return open(path, "rt", encoding="utf-8")

def normalize_chrom(chrom: str) -> str:
    c = chrom.strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    lc = c.lower()
    if lc in ("23", "x"): return "X"
    if lc in ("24", "y"): return "Y"
    if lc in ("25", "m", "mt"): return "MT"
    return c

# ---------- parse .psam (FID:=IID when FID missing) ----------

def parse_psam(psam_path: Path):
    samples, sex_codes = [], []
    with open_text(psam_path) as fh:
        header = fh.readline().strip().split()
        if not header:
            raise RuntimeError("Empty .psam")

        # Your files: ['#IID','SEX','SuperPop','Population','Project']
        iid_idx = header.index("#IID") if "#IID" in header else header.index("IID")
        sex_idx = header.index("SEX") if "SEX" in header else None

        for line in fh:
            s = line.strip()
            if not s: continue
            toks = s.split()
            iid = toks[iid_idx]
            fid = iid  # no FID column → FID := IID
            sx = toks[sex_idx] if (sex_idx is not None and sex_idx < len(toks)) else ""
            if sx in ("1","M","m","male","Male"): sex = 1
            elif sx in ("2","F","f","female","Female"): sex = 2
            else: sex = 0
            samples.append((fid, iid))
            sex_codes.append(sex)
    return samples, sex_codes

# ---------- iterate .pvar (.zst ok), yield only biallelic ----------

def pvar_rows(pvar_path: Path):
    with open_text(pvar_path) as fh:
        header = None
        vidx = 0
        for raw in fh:
            if raw.startswith("##"):  # meta
                continue
            if header is None:
                header = raw.strip().split()
                cols = {name: i for i, name in enumerate(header)}
                for need in ("#CHROM","POS","ID","REF","ALT"):
                    if need not in cols:
                        raise RuntimeError(f"Unexpected .pvar header: {header}")
                continue

            toks = raw.rstrip("\n").split()
            alt = toks[cols["ALT"]]
            cur = vidx
            vidx += 1
            if "," in alt:
                continue  # skip multiallelic for PLINK1 compatibility
            chrom = normalize_chrom(toks[cols["#CHROM"]])
            pos = int(toks[cols["POS"]])
            var_id = toks[cols["ID"]]
            ref = toks[cols["REF"]]
            yield cur, chrom, pos, var_id, ref, alt

# ---------- writers ----------

def write_fam(out_prefix: Path, samples, sex_codes):
    fam_path = out_prefix.with_suffix(".fam")
    fam_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fam_path, "w", encoding="utf-8") as out:
        for (fid, iid), sex in zip(samples, sex_codes):
            out.write(f"{fid} {iid} 0 0 {sex} -9\n")
    return fam_path

def write_bed_bim(pgen_path: Path, pvar_path: Path, out_prefix: Path, n_samples: int):
    bed_path = out_prefix.with_suffix(".bed")
    bim_path = out_prefix.with_suffix(".bim")
    with open(bed_path, "wb") as bed, open(bim_path, "w", encoding="utf-8") as bim:
        # PLINK .bed header: SNP-major mode
        bed.write(b"\x6C\x1B\x01")

        # pgenlib requires bytes path
        reader = pg.PgenReader(pgen_path.as_posix().encode("utf-8"))
        sample_ct = reader.get_raw_sample_ct()
        if sample_ct != n_samples:
            raise RuntimeError(f"Sample count mismatch: pgen={sample_ct}, psam={n_samples}")

        # int32 buffer for genotypes (ALT allele counts 0/1/2; negative for missing)
        geno = np.empty(sample_ct, dtype=np.int32)

        kept = 0
        for vidx, chrom, pos, var_id, ref, alt in pvar_rows(pvar_path):
            reader.read(vidx, geno, 1)  # allele_idx=1 → ALT counts

            # PLINK 1 two-bit codes per sample (SNP-major):
            #   00 => hom A1, 01 => missing, 10 => het, 11 => hom A2
            # we define A1=ALT, A2=REF in .bim
            codes = np.full(sample_ct, 0b01, dtype=np.uint8)  # start as missing
            codes[geno == 2] = 0b00  # hom ALT (A1/A1)
            codes[geno == 1] = 0b10  # het
            codes[geno == 0] = 0b11  # hom REF (A2/A2)

            # Pack 4 genotypes per byte (little-endian within the byte)
            full = (sample_ct // 4) * 4
            if full:
                m = codes[:full].reshape(-1, 4)
                blk = (m[:,0] | (m[:,1] << 2) | (m[:,2] << 4) | (m[:,3] << 6)).tobytes()
                bed.write(blk)
            if sample_ct > full:
                tail = codes[full:].tolist()
                while len(tail) < 4:
                    tail.append(0)  # padding (ignored)
                b = (tail[0] | (tail[1] << 2) | (tail[2] << 4) | (tail[3] << 6)) & 0xFF
                bed.write(bytes((b,)))

            # BIM row: CHR  ID  0  POS  A1(ALT)  A2(REF)
            snp_id = var_id if var_id != "." else f"{chrom}:{pos}:{ref}:{alt}"
            bim.write(f"{chrom}\t{snp_id}\t0\t{pos}\t{alt}\t{ref}\n")
            kept += 1

        reader.close()
    return bed_path, bim_path, kept

# ---------- main (no args) ----------

def main():
    here = Path(".").resolve()
    # Prefer GRCh38; fall back to GRCh37
    bases = ["GRCh38_HGDP+1kGP_ALL", "GRCh37_HGDP+1kGP_ALL"]
    base = None
    for b in bases:
        if (here / f"{b}.pgen").exists() and (here / f"{b}.psam").exists():
            base = b
            break
    if base is None:
        sys.exit("Could not find GRCh38_HGDP+1kGP_ALL or GRCh37_HGDP+1kGP_ALL in current directory.")

    out_prefix = here / ("hg38_plink1" if base.startswith("GRCh38") else "hg37_plink1")

    pgen = here / f"{base}.pgen"
    psam = here / f"{base}.psam"
    pvar = here / f"{base}.pvar"
    pvar_z = here / f"{base}.pvar.zst"
    pvar_path = pvar_z if pvar_z.exists() else pvar
    if not pvar_path.exists():
        sys.exit(f"Missing {pvar} (or {pvar_z})")

    print(f"[i] Input prefix:  {here / base}")
    print(f"[i] Output prefix: {out_prefix}")

    # FAM
    samples, sex_codes = parse_psam(psam)
    fam_path = write_fam(out_prefix, samples, sex_codes)
    print(f"[+] Wrote {fam_path.name} ({len(samples)} samples)")

    # BED/BIM
    bed_path, bim_path, kept = write_bed_bim(pgen, pvar_path, out_prefix, n_samples=len(samples))
    print(f"[✓] Wrote {bed_path.name}, {bim_path.name} (variants kept: {kept})")

if __name__ == "__main__":
    main()
