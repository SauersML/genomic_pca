from __future__ import annotations

import io
import os
import sys
import time
import shutil
from pathlib import Path

import numpy as np
import zstandard as zstd

import pgenlib as pg  # PLINK2 reader (Python API)

# ============================================================
# Pretty progress bars (TTY-aware)
# ============================================================

class ProgressBars:
    def __init__(self, enable: bool | None = None, width: int | None = None):
        self.enable = sys.stdout.isatty() if enable is None else enable
        self.tasks = []  # list of dicts: {name,total,cur,last_drawn,bar_width}
        self._last_redraw = 0.0
        self._printed_lines = 0
        self._term_width = (shutil.get_terminal_size().columns if self.enable else 80)
        # bar itself will fit inside width; leave room for label/percent/counts
        self._default_bar_width = max(10, min(40, self._term_width - 40)) if width is None else width

    def add(self, name: str, total: int, initial: int = 0, bar_width: int | None = None):
        bar_width = bar_width or self._default_bar_width
        self.tasks.append({
            "name": name,
            "total": max(1, int(total)),
            "cur": int(initial),
            "bar_width": int(bar_width),
            "last_drawn": -1  # force draw
        })

    def _format_line(self, t) -> str:
        total = t["total"]
        cur = min(max(0, t["cur"]), total)
        pct = (cur / total) * 100.0
        bar_w = t["bar_width"]
        filled = int(round((cur / total) * bar_w))
        bar = ("#" * filled) + ("-" * (bar_w - filled))
        return f"{t['name']:<6} [{bar}] {pct:6.2f}%  ({cur:,}/{total:,})"

    def _redraw(self, force: bool = False):
        if not self.enable:
            return
        now = time.time()
        # Throttle redraws to ~10Hz unless forced
        if not force and (now - self._last_redraw) < 0.1:
            return
        self._last_redraw = now

        # Move cursor up to the first progress line (if we've drawn before)
        if self._printed_lines:
            sys.stdout.write(f"\x1b[{self._printed_lines}F")  # move up N lines to start
        lines = []
        for t in self.tasks:
            lines.append(self._format_line(t))
        # Ensure lines don't overflow the terminal
        clipped = [ln[: self._term_width - 1] for ln in lines]
        sys.stdout.write("\n".join(clipped) + "\n")
        sys.stdout.flush()
        self._printed_lines = len(clipped)

    def update(self, name: str, cur: int | None = None, inc: int | None = None, force=False):
        for t in self.tasks:
            if t["name"] == name:
                if cur is not None:
                    t["cur"] = int(cur)
                if inc is not None:
                    t["cur"] += int(inc)
                # request redraw only when integer percent changed to minimize noise
                pct_now = int((t["cur"] / t["total"]) * 100)
                if pct_now != t["last_drawn"] or force:
                    t["last_drawn"] = pct_now
                    self._redraw(force=False)
                return
        raise KeyError(f"Unknown progress task: {name}")

    def finish(self, name: str):
        self.update(name, force=True)

    def finalize_all(self):
        # Draw final state one last time and then leave the bars on screen.
        if self.enable and self._printed_lines:
            self._redraw(force=True)


# ============================================================
# Helpers
# ============================================================

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


def human(n: int) -> str:
    return f"{n:,}"


# ============================================================
# PSAM parsing (FID := IID when FID col missing)
# ============================================================

def parse_psam(psam_path: Path):
    samples, sex_codes = [], []
    with open_text(psam_path) as fh:
        header = fh.readline().strip().split()
        if not header:
            raise RuntimeError("Empty .psam")

        # Typical: ['#IID','SEX','SuperPop','Population','Project']
        iid_idx = header.index("#IID") if "#IID" in header else header.index("IID")
        sex_idx = header.index("SEX") if "SEX" in header else None

        for line in fh:
            s = line.strip()
            if not s:
                continue
            toks = s.split()
            iid = toks[iid_idx]
            fid = iid  # no FID column → FID := IID
            sx = toks[sex_idx] if (sex_idx is not None and sex_idx < len(toks)) else ""
            if sx in ("1", "M", "m", "male", "Male"): sex = 1
            elif sx in ("2", "F", "f", "female", "Female"): sex = 2
            else: sex = 0
            samples.append((fid, iid))
            sex_codes.append(sex)
    return samples, sex_codes


# ============================================================
# PVAR iteration (yield only biallelic)
# ============================================================

def pvar_rows(pvar_path: Path):
    with open_text(pvar_path) as fh:
        header = None
        vidx = 0
        for raw in fh:
            if raw.startswith("##"):
                continue
            if header is None:
                header = raw.strip().split()
                cols = {name: i for i, name in enumerate(header)}
                for need in ("#CHROM", "POS", "ID", "REF", "ALT"):
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


def count_biallelic_variants(pvar_path: Path) -> int:
    cnt = 0
    for _ in pvar_rows(pvar_path):
        cnt += 1
    return cnt


# ============================================================
# Writers with resume support
# ============================================================

def write_fam(out_prefix: Path, samples, sex_codes, pbar: ProgressBars):
    fam_path = out_prefix.with_suffix(".fam")
    fam_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(samples)
    # If complete .fam already exists with expected number of lines, keep it
    existing = fam_path.exists() and sum(1 for _ in open(fam_path, "rb")) == total

    pbar.add("FAM", total, initial=(total if existing else 0))
    if existing:
        pbar.finish("FAM")
        return fam_path

    with open(fam_path, "w", encoding="utf-8") as out:
        # write line-by-line to show progress
        for i, ((fid, iid), sex) in enumerate(zip(samples, sex_codes), start=1):
            out.write(f"{fid} {iid} 0 0 {sex} -9\n")
            pbar.update("FAM", cur=i)
    pbar.finish("FAM")
    return fam_path


def _bytes_per_snp(sample_ct: int) -> int:
    # SNP-major .bed packs 4 two-bit genotypes per byte
    return (sample_ct + 3) // 4


def _safe_truncate(fpath: Path, size: int):
    with open(fpath, "r+b") as f:
        f.truncate(size)


def _truncate_bed_to_boundary(bed_path: Path, bps: int) -> int:
    """
    Ensure .bed is aligned to a full SNP boundary.
    Returns the number of *complete* variants currently stored.
    """
    sz = bed_path.stat().st_size
    if sz < 3:
        if sz > 0:
            _safe_truncate(bed_path, 0)
        return 0
    rem = (sz - 3) % bps
    if rem:
        _safe_truncate(bed_path, sz - rem)
        sz -= rem
    return (sz - 3) // bps


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _rebuild_bim_prefix(bim_path: Path, pvar_path: Path, n_rows: int):
    # Recreate the first n_rows lines of BIM from PVAR (no PGEN reads needed)
    with open(bim_path, "w", encoding="utf-8") as out_bim:
        cnt = 0
        for _, chrom, pos, var_id, ref, alt in pvar_rows(pvar_path):
            snp_id = var_id if var_id != "." else f"{chrom}:{pos}:{ref}:{alt}"
            out_bim.write(f"{chrom}\t{snp_id}\t0\t{pos}\t{alt}\t{ref}\n")
            cnt += 1
            if cnt == n_rows:
                break


def write_bed_bim_resume(pgen_path: Path, pvar_path: Path, out_prefix: Path,
                         n_samples: int, pbar: ProgressBars):
    bed_path = out_prefix.with_suffix(".bed")
    bim_path = out_prefix.with_suffix(".bim")

    # Open PGEN to confirm sample count and compute bytes-per-SNP
    reader = pg.PgenReader(pgen_path.as_posix().encode("utf-8"))
    sample_ct = reader.get_raw_sample_ct()
    if sample_ct != n_samples:
        reader.close()
        raise RuntimeError(f"Sample count mismatch: pgen={sample_ct}, psam={n_samples}")
    bps = _bytes_per_snp(sample_ct)

    # Determine total biallelic variants to emit (accurate % requires this)
    total_vars = count_biallelic_variants(pvar_path)

    # --- Determine 'done' (how many biallelic variants already emitted) ---
    done = 0
    if bed_path.exists():
        done = _truncate_bed_to_boundary(bed_path, bps)

    # BIM sanity & alignment with BED
    if bim_path.exists():
        bim_lines = _count_lines(bim_path)
        if bim_lines != done:
            keep = min(done, bim_lines)
            if keep != done:
                # Trim BED down to BIM (more conservative)
                _safe_truncate(bed_path, 3 + keep * bps) if bed_path.exists() else None
                done = keep
            # Rebuild BIM prefix to exactly 'done' lines
            _rebuild_bim_prefix(bim_path, pvar_path, done)
    else:
        # Missing BIM: create the first 'done' lines
        _rebuild_bim_prefix(bim_path, pvar_path, done)

    # --- Open outputs ---
    if bed_path.exists() and done > 0:
        bed = open(bed_path, "ab")
        # Header sanity
        with open(bed_path, "rb") as h:
            if h.read(3) != b"\x6C\x1B\x01":
                bed.close(); reader.close()
                raise RuntimeError("Existing .bed header is invalid.")
        bim = open(bim_path, "a", encoding="utf-8")
    else:
        # fresh start
        bed = open(bed_path, "wb")
        bed.write(b"\x6C\x1B\x01")
        bim = open(bim_path, "w", encoding="utf-8")
        done = 0  # ensure consistency

    # --- Progress bars for BED and BIM (per file) ---
    pbar.add("BED", total_vars, initial=done)
    pbar.add("BIM", total_vars, initial=done)

    # Work buffer for genotypes (ALT allele counts 0/1/2; negative for missing)
    geno = np.empty(sample_ct, dtype=np.int32)

    # Stream through PVAR; skip already-finished biallelic variants
    try:
        processed = 0
        for i, (vidx, chrom, pos, var_id, ref, alt) in enumerate(pvar_rows(pvar_path)):
            if i < done:
                continue  # already emitted
            # Read ALT counts for this PGEN variant index
            reader.read(vidx, geno, 1)

            # PLINK 1 two-bit codes per sample (SNP-major):
            #   00 => hom A1, 01 => missing, 10 => het, 11 => hom A2
            # We define A1=ALT, A2=REF in .bim
            codes = np.full(sample_ct, 0b01, dtype=np.uint8)  # start as missing
            codes[geno == 2] = 0b00  # hom ALT
            codes[geno == 1] = 0b10  # het
            codes[geno == 0] = 0b11  # hom REF

            # Pack 4 genotypes per byte (little-endian within the byte)
            full = (sample_ct // 4) * 4
            if full:
                m = codes[:full].reshape(-1, 4)
                blk = (m[:, 0] | (m[:, 1] << 2) | (m[:, 2] << 4) | (m[:, 3] << 6)).tobytes()
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

            processed += 1
            cur = done + processed
            # Update both file progress bars
            pbar.update("BED", cur=cur)
            pbar.update("BIM", cur=cur)

    except KeyboardInterrupt:
        # Graceful interruption: ensure .bed stays aligned (we only ever write whole SNPs),
        # and .bim lines always match. Nothing to do here except close files.
        print("\n[!] Interrupted by user. Partial outputs left in a consistent state for resume.", file=sys.stderr)
        raise
    finally:
        bed.close()
        bim.close()
        reader.close()

    pbar.finish("BED")
    pbar.finish("BIM")

    kept = done + processed
    return bed_path, bim_path, kept, total_vars


# ============================================================
# MAIN
# ============================================================

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

    # TTY-aware progress display
    pbar = ProgressBars(enable=None)

    # FAM
    samples, sex_codes = parse_psam(psam)
    fam_path = write_fam(out_prefix, samples, sex_codes, pbar)
    print(f"[+] {fam_path.name}: {human(len(samples))} samples")

    # BED/BIM with resume
    try:
        bed_path, bim_path, kept, total = write_bed_bim_resume(
            pgen, pvar_path, out_prefix, n_samples=len(samples), pbar=pbar
        )
    finally:
        pbar.finalize_all()

    print(f"[✓] {bed_path.name}, {bim_path.name}: variants written = {human(kept)} / {human(total)}")
    if kept < total:
        print("[→] You can safely rerun this script to resume later; it will append remaining variants.")

if __name__ == "__main__":
    # On non-TTY environments, ensure prints are flushed regularly
    try:
        main()
    except KeyboardInterrupt:
        # Make exit status clear on Ctrl-C
        sys.exit(130)
