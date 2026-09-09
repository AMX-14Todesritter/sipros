#!/usr/bin/env python3
"""Create dependency-free SVG explanations for one peak_export result directory."""
import csv
import html
import sys
from pathlib import Path


def read_tsv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def esc(value):
    return html.escape(str(value))


def save_svg(path, body, width=1400, height=760):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fbfaf7"/>
<style>
text {{ font-family: Arial, "Noto Sans", sans-serif; fill:#24313b }}
.title {{ font-size:28px; font-weight:700 }} .sub {{ font-size:15px; fill:#596873 }}
.axis {{ stroke:#596873; stroke-width:1.5 }} .grid {{ stroke:#dfe5e8; stroke-width:1 }}
.tick {{ font-size:13px; fill:#596873 }} .note {{ font-size:17px }}
</style>{body}</svg>'''
    path.write_text(svg)


def spectrum_svg(rows, path, title, subtitle, value_key="intensity", mz_key="mz", color="#247ba0"):
    width, height = 1400, 760
    left, right, top, bottom = 90, 40, 115, 90
    mz = [float(r[mz_key]) for r in rows]
    vals = [float(r[value_key]) for r in rows]
    xmin, xmax = min(mz), max(mz)
    vmax = max(vals) or 1
    x = lambda v: left + (v-xmin)/(xmax-xmin)*(width-left-right)
    y = lambda v: height-bottom-v/vmax*(height-top-bottom)
    parts = [f'<text x="{left}" y="42" class="title">{esc(title)}</text>',
             f'<text x="{left}" y="72" class="sub">{esc(subtitle)}</text>']
    for i in range(6):
        yy = top + i*(height-top-bottom)/5
        val = vmax*(1-i/5)
        parts += [f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" class="grid"/>',
                  f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" class="tick">{val:.0f}</text>']
    for i in range(9):
        xx = left + i*(width-left-right)/8
        val = xmin + i*(xmax-xmin)/8
        parts += [f'<line x1="{xx:.1f}" y1="{height-bottom}" x2="{xx:.1f}" y2="{height-bottom+7}" class="axis"/>',
                  f'<text x="{xx:.1f}" y="{height-bottom+28}" text-anchor="middle" class="tick">{val:.0f}</text>']
    parts += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>',
              f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>',
              f'<text x="{(left+width-right)/2:.0f}" y="{height-25}" text-anchor="middle" class="note">m/z</text>']
    for m, v in zip(mz, vals):
        parts.append(f'<line x1="{x(m):.2f}" y1="{height-bottom}" x2="{x(m):.2f}" y2="{y(v):.2f}" stroke="{color}" stroke-width="2"/>')
    save_svg(path, "".join(parts), width, height)


def mvh_svg(rows, path, scan):
    colors = {1:"#d62828", 2:"#f77f00", 3:"#457b9d", 4:"#9aa6ad"}
    width, height = 1400, 760
    left, right, top, bottom = 90, 40, 125, 90
    mz = [float(r["mz"]) for r in rows]
    xmin, xmax = min(mz), max(mz)
    x = lambda v: left + (v-xmin)/(xmax-xmin)*(width-left-right)
    parts = [f'<text x="{left}" y="42" class="title">MVH-retained peaks — scan {scan}</text>',
             f'<text x="{left}" y="72" class="sub">Height and color encode MVH intensity class; class 1 is the strongest class.</text>']
    baseline = height-bottom
    heights = {1:500, 2:350, 3:210, 4:100}
    for r in rows:
        c = int(r["mvh_class"]); xx=x(float(r["mz"])); yy=baseline-heights.get(c, 70)
        parts.append(f'<line x1="{xx:.2f}" y1="{baseline}" x2="{xx:.2f}" y2="{yy}" stroke="{colors.get(c,"#999")}" stroke-width="3"/>')
    parts += [f'<line x1="{left}" y1="{baseline}" x2="{width-right}" y2="{baseline}" class="axis"/>']
    for i in range(9):
        xx=left+i*(width-left-right)/8; val=xmin+i*(xmax-xmin)/8
        parts += [f'<text x="{xx:.1f}" y="{baseline+28}" text-anchor="middle" class="tick">{val:.0f}</text>']
    for idx, c in enumerate(sorted(set(int(r["mvh_class"]) for r in rows))):
        xx=left+idx*190
        parts += [f'<rect x="{xx}" y="88" width="18" height="18" fill="{colors.get(c,"#999")}"/>',
                  f'<text x="{xx+28}" y="103" class="tick">Class {c}</text>']
    parts.append(f'<text x="{(left+width-right)/2:.0f}" y="{height-25}" text-anchor="middle" class="note">m/z</text>')
    save_svg(path, "".join(parts), width, height)


def candidate_svg(row, path):
    fields = [("Scan", row["scan_id"]), ("Precursor charge", row["precursor_charge"]),
              ("Candidate peptide", row["peptide"]), ("Scoring sequence", row["scoring_sequence"]),
              ("Protein source", row["protein_source"]), ("Peptide mass", f'{float(row["peptide_mass"]):.6f} Da'),
              ("Fragment tolerance", f'±{float(row["fragment_tolerance"]):g} Da')]
    parts = ['<text x="80" y="55" class="title">Candidate peptide information card</text>',
             '<text x="80" y="85" class="sub">This is the first mass-window candidate reaching MVH scoring, not necessarily the best identification.</text>',
             '<rect x="70" y="120" width="1260" height="550" rx="22" fill="#ffffff" stroke="#cad5da" stroke-width="2"/>']
    for i,(key,val) in enumerate(fields):
        yy=175+i*66
        parts += [f'<text x="115" y="{yy}" class="sub">{esc(key)}</text>',
                  f'<text x="390" y="{yy}" font-size="21" font-weight="600">{esc(val)}</text>']
        if i < len(fields)-1: parts.append(f'<line x1="110" y1="{yy+22}" x2="1290" y2="{yy+22}" class="grid"/>')
    save_svg(path, "".join(parts), 1400, 740)


def overlay_svg(obs, theo, tol, path, scan):
    width,height=1400,800; left,right,top,bottom=90,40,125,90
    mz=[float(r["mz"]) for r in obs]; inten=[float(r["intensity"]) for r in obs]
    tmz=[float(r["theoretical_mz"]) for r in theo]
    xmin=min(min(mz),min(tmz)); xmax=max(max(mz),max(tmz)); vmax=max(inten)
    x=lambda v:left+(v-xmin)/(xmax-xmin)*(width-left-right)
    y=lambda v:height-bottom-v/vmax*(height-top-bottom)
    matches=[]
    for t in tmz:
        candidates=[(abs(m-t),m,i) for m,i in zip(mz,inten) if abs(m-t) < tol]
        if candidates: matches.append((t,min(candidates)))
    parts=[f'<text x="{left}" y="42" class="title">Observed vs theoretical peaks — scan {scan}</text>',
           f'<text x="{left}" y="72" class="sub">Blue: observed intensity. Orange: theoretical fragments. Green circles: matches within ±{tol:g} Da ({len(matches)}/{len(tmz)}).</text>']
    base=height-bottom
    for m,i in zip(mz,inten): parts.append(f'<line x1="{x(m):.2f}" y1="{base}" x2="{x(m):.2f}" y2="{y(i):.2f}" stroke="#3a86a8" stroke-width="2" opacity=".75"/>')
    for idx,t in enumerate(tmz):
        xx=x(t); h=80+(idx%2)*28
        parts += [f'<line x1="{xx:.2f}" y1="{base}" x2="{xx:.2f}" y2="{base-h}" stroke="#f28e2b" stroke-width="3" stroke-dasharray="7 4"/>',
                  f'<text x="{xx:.2f}" y="{base-h-7}" text-anchor="middle" class="tick">T{idx}</text>']
    for t,(_,m,i) in matches: parts.append(f'<circle cx="{x(m):.2f}" cy="{y(i):.2f}" r="8" fill="none" stroke="#2a9d55" stroke-width="4"/>')
    parts.append(f'<line x1="{left}" y1="{base}" x2="{width-right}" y2="{base}" class="axis"/>')
    for i in range(9):
        xx=left+i*(width-left-right)/8; val=xmin+i*(xmax-xmin)/8
        parts.append(f'<text x="{xx:.1f}" y="{base+28}" text-anchor="middle" class="tick">{val:.0f}</text>')
    parts.append(f'<text x="{(left+width-right)/2:.0f}" y="{height-25}" text-anchor="middle" class="note">m/z</text>')
    save_svg(path,"".join(parts),width,height)
    return matches


def main():
    root=Path(sys.argv[1] if len(sys.argv)>1 else ".").resolve(); out=root/"figures"; out.mkdir(exist_ok=True)
    obs=read_tsv(root/"observed_unsorted.tsv"); mvh=read_tsv(root/"observed_mvh.tsv")
    theo=read_tsv(root/"theoretical_unsorted.tsv"); cand=read_tsv(root/"candidate.tsv")[0]
    scan=cand["scan_id"]; tol=float(cand["fragment_tolerance"])
    spectrum_svg(obs,out/"01_observed_unsorted.svg",f"Raw observed spectrum — scan {scan}","Each stick is one measured peak; height is raw intensity.")
    mvh_svg(mvh,out/"02_observed_mvh.svg",scan)
    inrange=[r for r in theo if r["in_spectrum_range"]=="1"]
    spectrum_svg(inrange,out/"03_theoretical_unsorted.svg",f"Theoretical fragment spectrum — {cand['peptide']}","Equal-height sticks show predicted fragment m/z in generation order; no theoretical intensities are available.",value_key="in_spectrum_range",mz_key="theoretical_mz",color="#f28e2b")
    candidate_svg(cand,out/"04_candidate.svg")
    matches=overlay_svg(obs,theo,tol,out/"05_observed_theoretical_overlay.svg",scan)
    print(f"Created 5 SVG files in {out}; tolerance matches: {len(matches)}/{len(theo)}")


if __name__ == "__main__": main()
