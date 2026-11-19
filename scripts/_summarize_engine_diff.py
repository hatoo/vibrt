"""Parse RESULT lines from `_run_all_scenes_nocomp_exr.sh`'s log and emit
a clean Markdown table.

Usage:
  py scripts/_summarize_engine_diff.py /tmp/all_scenes_nocomp.log
"""
import re
import sys
from pathlib import Path

PAT = re.compile(
    r"RESULT tag=(?P<tag>\S+) spp=(?P<spp>\d+) pct=(?P<pct>\d+) "
    r"H=(?P<H>\d+) W=(?P<W>\d+) "
    r"v_mean=\((?P<vR>[^,]+),(?P<vG>[^,]+),(?P<vB>[^)]+)\) "
    r"c_mean=\((?P<cR>[^,]+),(?P<cG>[^,]+),(?P<cB>[^)]+)\) "
    r"abs_diff=(?P<adiff>\S+) ratio=(?P<ratio>\S+) "
    r"v_p99=(?P<vp99>\S+) c_p99=(?P<cp99>\S+) "
    r"v_peak=(?P<vpeak>\S+) c_peak=(?P<cpeak>\S+)"
)


def main(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    rows = []
    for m in PAT.finditer(text):
        rows.append(m.groupdict())

    if not rows:
        print("No RESULT lines parsed.")
        return

    spp_set = {r["spp"] for r in rows}
    pct_set = {r["pct"] for r in rows}
    print(f"\n# EXR-based engine accuracy (compositor=OFF, view=Raw)")
    print(f"\nSPP={','.join(sorted(spp_set))}, PCT={','.join(sorted(pct_set))}")
    print()
    print("| scene | resolution | vibrt mean | cycles mean | |diff| | ratio v/c | v_peak | c_peak |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        vmean = f"({float(r['vR']):.3f},{float(r['vG']):.3f},{float(r['vB']):.3f})"
        cmean = f"({float(r['cR']):.3f},{float(r['cG']):.3f},{float(r['cB']):.3f})"
        adiff = float(r["adiff"])
        ratio = float(r["ratio"])
        vp = float(r["vpeak"]); cp = float(r["cpeak"])
        marker = ""
        if abs(ratio - 1.0) > 0.20:
            marker = " [!!]"
        elif abs(ratio - 1.0) > 0.05:
            marker = " [!]"
        print(f"| {r['tag']} | {r['W']}x{r['H']} | {vmean} | {cmean} | "
              f"{adiff:.4f} | {ratio:.3f}{marker} | {vp:.1f} | {cp:.1f} |")

    print("\n[!!] = ratio differs from 1.0 by >20%, [!] = >5%")
    print(f"\nrows: {len(rows)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/all_scenes_nocomp.log")
