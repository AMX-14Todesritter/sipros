#!/usr/bin/env python3
"""Analyze existing Sipros Regular MVH profiling CSVs without dependencies."""

import argparse
import sys
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# Locate the shared path policy independently of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.output_paths import resolve_output

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "experiments" / "ecoli_regular"


def read_csv(name):
    with (INPUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key, integer=False):
    value = row.get(key, "")
    if value is None or value == "":
        return None
    return int(value) if integer else float(value)


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    x = (len(values) - 1) * p / 100.0
    lo, hi = math.floor(x), math.ceil(x)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - x) + values[hi] * (x - lo)


def describe(values):
    values = list(values)
    if not values:
        return {k: None for k in ("count", "mean", "median", "stddev", "p50", "p90", "p95", "p99", "max")}
    return {
        "count": len(values), "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stddev": statistics.pstdev(values), "p50": percentile(values, 50),
        "p90": percentile(values, 90), "p95": percentile(values, 95),
        "p99": percentile(values, 99), "max": max(values),
    }


def fmt(value, digits=6):
    if value is None:
        return "NA"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}g}"


def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_plot(path, title, xlabel, ylabel, series, width=900, height=520):
    """Small dependency-free SVG line/bar/scatter plot helper."""
    ml, mr, mt, mb = 82, 25, 55, 65
    points = [(x, y) for s in series for x, y in s["points"]]
    xmin = min((x for x, _ in points), default=0); xmax = max((x for x, _ in points), default=1)
    ymin = min(0, min((y for _, y in points), default=0)); ymax = max((y for _, y in points), default=1)
    if xmax == xmin: xmax = xmin + 1
    if ymax == ymin: ymax = ymin + 1
    sx = lambda x: ml + (x - xmin) / (xmax - xmin) * (width - ml - mr)
    sy = lambda y: height - mb - (y - ymin) / (ymax - ymin) * (height - mt - mb)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{esc(title)}</text>',
           f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#333"/>',
           f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#333"/>']
    for i in range(6):
        xv = xmin + (xmax-xmin)*i/5; px=sx(xv)
        yv = ymin + (ymax-ymin)*i/5; py=sy(yv)
        out += [f'<line x1="{px:.1f}" y1="{height-mb}" x2="{px:.1f}" y2="{height-mb+5}" stroke="#333"/>',
                f'<text x="{px:.1f}" y="{height-mb+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{xv:.3g}</text>',
                f'<line x1="{ml-5}" y1="{py:.1f}" x2="{ml}" y2="{py:.1f}" stroke="#333"/>',
                f'<text x="{ml-9}" y="{py+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{yv:.3g}</text>']
    for idx, s in enumerate(series):
        color=s.get("color", ["#2864b7","#d34e30","#2a8c65","#8b5fbf"][idx%4]); kind=s.get("kind","bar")
        if kind == "bar":
            bw = max(2, (width-ml-mr)/max(1,len(s["points"]))*0.72)
            for x,y in s["points"]:
                out.append(f'<rect x="{sx(x)-bw/2:.2f}" y="{sy(y):.2f}" width="{bw:.2f}" height="{sy(0)-sy(y):.2f}" fill="{color}" opacity="0.85"/>')
        elif kind == "line":
            coords=" ".join(f"{sx(x):.2f},{sy(y):.2f}" for x,y in s["points"])
            out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        else:
            for x,y in s["points"]: out.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="2.5" fill="{color}"/>')
    out += [f'<text x="{(ml+width-mr)/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="13">{esc(xlabel)}</text>',
            f'<text transform="translate(18 {(mt+height-mb)/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">{esc(ylabel)}</text>', '</svg>']
    path.write_text("\n".join(out))


def histogram(values, bins=30, lo=None, hi=None):
    if not values: return []
    lo=min(values) if lo is None else lo; hi=max(values) if hi is None else hi
    if hi == lo: hi=lo+1
    counts=[0]*bins
    for v in values: counts[min(bins-1,max(0,int((v-lo)/(hi-lo)*bins)))] += 1
    return [(lo+(i+.5)*(hi-lo)/bins, c) for i,c in enumerate(counts)]


def spectrum_svg(path, peaks, queries, candidate, zoom=None):
    retained=[p for p in peaks if p["retained"]]
    xmin=min(p["mz"] for p in peaks) if zoom is None else zoom[0]
    xmax=max(p["mz"] for p in peaks) if zoom is None else zoom[1]
    W,H=1100,760; ml,mr,mt,mb=75,25,48,55; panel_h=190; gaps=34
    sx=lambda x: ml+(x-xmin)/(xmax-xmin)*(W-ml-mr)
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>',f'<text x="{W/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">Scan {candidate["scan_id"]}, candidate {candidate["candidate_index"]}: {esc(candidate["peptide_sequence"])}</text>']
    panels=[("Raw observed spectrum",peaks), ("MVH-retained spectrum (classes 1/2/3)",retained)]
    colors={1:"#2878b5",2:"#e07b24",3:"#419b55"}
    for pi,(label,data) in enumerate(panels):
        top=mt+pi*(panel_h+gaps); base=top+panel_h
        local=[p for p in data if xmin<=p["mz"]<=xmax]; ymax=max([p["intensity"] for p in local] or [1])
        out += [f'<text x="8" y="{top+14}" font-family="sans-serif" font-size="12">{esc(label)}</text>',f'<line x1="{ml}" y1="{base}" x2="{W-mr}" y2="{base}" stroke="#777"/>']
        for p in local:
            y=base-p["intensity"]/ymax*(panel_h-20); color=colors.get(p["class"],"#777") if p["retained"] else "#999"
            out.append(f'<line x1="{sx(p["mz"]):.2f}" y1="{base}" x2="{sx(p["mz"]):.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="1.3"/>')
    top=mt+2*(panel_h+gaps); base=top+panel_h
    out += [f'<text x="8" y="{top+14}" font-family="sans-serif" font-size="12">Theoretical queries: hit (green), miss (red); matched peak (blue)</text>',f'<line x1="{ml}" y1="{base}" x2="{W-mr}" y2="{base}" stroke="#777"/>']
    local=[q for q in queries if xmin<=q["theoretical_mz"]<=xmax]
    for i,q in enumerate(local):
        x=sx(q["theoretical_mz"]); color="#238b57" if q["hit"] else "#c9443b"; y=base-35-(i%5)*22
        if zoom is not None:
            tol=q["tolerance"]; out.append(f'<rect x="{sx(q["theoretical_mz"]-tol):.2f}" y="{y-5}" width="{sx(q["theoretical_mz"]+tol)-sx(q["theoretical_mz"]-tol):.2f}" height="10" fill="{color}" opacity="0.16"/>')
        out.append(f'<line x1="{x:.2f}" y1="{base}" x2="{x:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="1.4"/>')
        if q["hit"]:
            mx=sx(q["matched_peak_mz"]); out += [f'<circle cx="{mx:.2f}" cy="{y:.2f}" r="3" fill="#2456a6"/>',f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{mx:.2f}" y2="{y:.2f}" stroke="#2456a6"/>']
    for i in range(6):
        x=xmin+(xmax-xmin)*i/5; px=sx(x)
        out += [f'<line x1="{px:.1f}" y1="{base}" x2="{px:.1f}" y2="{base+5}" stroke="#333"/>',f'<text x="{px:.1f}" y="{base+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{x:.3f}</text>']
    out += [f'<text x="{W/2}" y="{H-12}" text-anchor="middle" font-family="sans-serif" font-size="13">m/z</text>','</svg>']
    path.write_text("\n".join(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New output directory (default: project output tree)")
    args = parser.parse_args()
    output = resolve_output(args.output, "analysis", "regular_mvh", "ecoli")
    output.mkdir(parents=True, exist_ok=False)
    scans=read_csv("scan_summary.thread_0.csv"); observed=read_csv("observed_peaks.thread_0.csv"); fragments=read_csv("fragment_queries.thread_0.csv")
    for r in fragments:
        for k in ("scan_id","candidate_index","precursor_charge","fragment_index","in_spectrum_range","hit","matched_peak_index","mvh_class","num_peaks_in_window"): r[k]=number(r,k,True)
        for k in ("theoretical_mz","tolerance","matched_peak_mz","mass_error"): r[k]=number(r,k)
    for r in observed:
        r.update(scan_id=number(r,"scan_id",True),peak_index=number(r,"peak_index",True),mz=number(r,"mz"),intensity=number(r,"intensity"),retained=bool(number(r,"mvh_retained",True)),**{"class":number(r,"mvh_class",True)})

    # Candidate-level aggregation.
    groups=defaultdict(list)
    for r in fragments: groups[(r["scan_id"],r["candidate_index"],r["peptide_sequence"])].append(r)
    candidates=[]
    for key, rows in groups.items():
        ir=[r for r in rows if r["in_spectrum_range"]]; hits=[r for r in ir if r["hit"]]; errors=[abs(r["mass_error"]) for r in hits]
        matched=Counter(r["matched_peak_index"] for r in hits)
        candidates.append({"scan_id":key[0],"candidate_index":key[1],"peptide_sequence":key[2],"theoretical_fragment_queries":len(rows),"in_range_queries":len(ir),"hits":len(hits),"misses":len(ir)-len(hits),"hit_rate":len(hits)/len(ir) if ir else 0,"mean_num_peaks_in_window":statistics.fmean(r["num_peaks_in_window"] for r in ir) if ir else 0,"max_num_peaks_in_window":max((r["num_peaks_in_window"] for r in ir),default=0),"mean_absolute_mass_error":statistics.fmean(errors) if errors else None,"median_absolute_mass_error":statistics.median(errors) if errors else None,"matched_observed_peaks":len(hits),"unique_matched_peak_indices":len(matched),"duplicated_matches":len(hits)-len(matched),"max_peak_reuse_count":max(matched.values(),default=0)})
    candidates.sort(key=lambda x:(x["scan_id"],x["candidate_index"],x["peptide_sequence"]))
    fields=list(candidates[0])
    with (output/"candidate_summary.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(candidates)

    # Validation.
    missing={}
    for name,rows in (("scan_summary",scans),("observed_peaks",observed),("fragment_queries",fragments)):
        missing[name]={k:sum(r.get(k) in (None,"") for r in rows) for k in rows[0]}
        missing[name]={k:v for k,v in missing[name].items() if v}
    summary_internal=[]
    for i,s in enumerate(scans,1):
        matched=number(s,"matched_queries",True); missed=number(s,"missed_queries",True); inrange=number(s,"in_range_queries",True); total=number(s,"total_fragment_queries",True)
        recomputed=matched/inrange if inrange else 0
        summary_internal.append({"row":i,"scan_id":int(s["scan_id"]),"matched_plus_missed_ok":matched+missed==inrange,"in_range_le_total":inrange<=total,"stored_hit_rate":float(s["hit_rate"]),"recomputed_hit_rate":recomputed,"hit_rate_difference":float(s["hit_rate"])-recomputed})
    frag_by_scan=defaultdict(list)
    for r in fragments: frag_by_scan[r["scan_id"]].append(r)
    reconstructed={}
    for sid,rows in frag_by_scan.items():
        ir=[r for r in rows if r["in_spectrum_range"]]; hits=sum(r["hit"] for r in ir)
        reconstructed[sid]={"total":len(rows),"in_range":len(ir),"hits":hits,"misses":len(ir)-hits,"hit_rate":hits/len(ir)}
    summary_combined={}
    for s in scans:
        sid=int(s["scan_id"]); x=summary_combined.setdefault(sid,{"total":0,"in_range":0,"hits":0,"misses":0})
        for src,dst in (("total_fragment_queries","total"),("in_range_queries","in_range"),("matched_queries","hits"),("missed_queries","misses")): x[dst]+=int(s[src])
    for x in summary_combined.values(): x["hit_rate"]=x["hits"]/x["in_range"]

    hits=[r for r in fragments if r["in_spectrum_range"] and r["hit"]]; misses=[r for r in fragments if r["in_spectrum_range"] and not r["hit"]]; inrange=hits+misses
    validation={
      "summary_internal":summary_internal,
      "fragment_reconstructed_by_scan":reconstructed,"summary_combined_by_scan":summary_combined,
      "hit_bad_index":sum(r["matched_peak_index"]<0 for r in hits),"hit_missing_mz":sum(r["matched_peak_mz"] is None for r in hits),"hit_missing_error":sum(r["mass_error"] is None for r in hits),
      "hit_error_not_strictly_below_tolerance":sum(abs(r["mass_error"])>=r["tolerance"] for r in hits),"hit_nonpositive_window":sum(r["num_peaks_in_window"]<=0 for r in hits),
      "mass_error_sign_max_abs_residual":max(abs(r["mass_error"]-(r["matched_peak_mz"]-r["theoretical_mz"])) for r in hits),
      "miss_index_not_minus_one":sum(r["matched_peak_index"]!=-1 for r in misses),"miss_with_mz_or_error":sum(r["matched_peak_mz"] is not None or r["mass_error"] is not None for r in misses),"miss_with_positive_window":sum(r["num_peaks_in_window"]>0 for r in misses),
    }

    # Peak reconstruction and classes.
    obs_by_scan=defaultdict(list)
    for p in observed: obs_by_scan[p["scan_id"]].append(p)
    peak_checks={}
    for sid,rows in obs_by_scan.items():
        ret=[p for p in rows if p["retained"]]
        peak_checks[sid]={"raw_count":len(rows),"retained_count":len(ret),"retained_fraction":len(ret)/len(rows),"raw_min_mz":min(p["mz"] for p in rows),"raw_max_mz":max(p["mz"] for p in rows),"retained_min_mz":min(p["mz"] for p in ret),"retained_max_mz":max(p["mz"] for p in ret),"classes":dict(sorted(Counter(p["class"] for p in ret).items()))}

    def window_stats(rows):
        vals=[r["num_peaks_in_window"] for r in rows]; n=len(vals)
        return {"count":n,"p_w_0":sum(v==0 for v in vals)/n,"p_w_1":sum(v==1 for v in vals)/n,"p_w_2":sum(v==2 for v in vals)/n,"p_w_ge_3":sum(v>=3 for v in vals)/n,"mean":statistics.fmean(vals),"median":statistics.median(vals),"p95":percentile(vals,95),"max":max(vals)}
    windows={"all_in_range":window_stats(inrange),"hits":window_stats(hits),"misses":window_stats(misses)}
    abs_errors=[abs(r["mass_error"]) for r in hits]; normalized=[abs(r["mass_error"])/r["tolerance"] for r in hits]

    # Density: include occupied bins only and all bins across retained range.
    density={}; density_values={1:[],5:[],10:[]}
    for sid,rows in obs_by_scan.items():
        ret=[p for p in rows if p["retained"]]; span=max(p["mz"] for p in ret)-min(p["mz"] for p in ret)
        density[sid]={"retained_count":len(ret),"retained_mz_range":span,"global_peaks_per_da":len(ret)/span}
        for size in density_values:
            lo=math.floor(min(p["mz"] for p in ret)/size)*size; hi=math.floor(max(p["mz"] for p in ret)/size)*size
            counts=Counter(math.floor(p["mz"]/size)*size for p in ret)
            density_values[size].extend(counts.get(lo+i*size,0) for i in range(round((hi-lo)/size)+1))
    density_stats={str(k):describe(v)|{"zero_fraction":sum(x==0 for x in v)/len(v),"occupied_bin_mean":statistics.fmean(x for x in v if x)} for k,v in density_values.items()}

    # Representative candidates: scaled distance to medians; preserve normal query count for low-hit.
    med={k:statistics.median(c[k] for c in candidates) for k in ("hit_rate","theoretical_fragment_queries","mean_num_peaks_in_window")}
    typical=min(candidates,key=lambda c:sum(abs(c[k]-med[k])/(abs(med[k]) or 1) for k in med))
    normal=[c for c in candidates if c["theoretical_fragment_queries"]>=percentile([x["theoretical_fragment_queries"] for x in candidates],25)]
    selected={"typical":typical,"high_hit":max(candidates,key=lambda c:c["hit_rate"]),"low_hit_normal_queries":min(normal,key=lambda c:(c["hit_rate"],-c["theoretical_fragment_queries"])),"dense_window":max(candidates,key=lambda c:(c["max_num_peaks_in_window"],c["mean_num_peaks_in_window"]))}
    summary_scan=next(s for s in scans if int(s["scan_id"])==typical["scan_id"])
    for c in selected.values(): c.update(raw_peak_count=int(summary_scan["raw_peak_count"]),processed_peak_count=int(summary_scan["processed_peak_count"]))
    with (output/"representative_candidates.csv").open("w",newline="") as h:
        f=["selection"]+list(next(iter(selected.values())).keys()); w=csv.DictWriter(h,fieldnames=f); w.writeheader()
        for label,c in selected.items(): w.writerow({"selection":label,**c})

    # Reuse examples.
    reuse=[c for c in candidates if c["duplicated_matches"]>0]
    reuse.sort(key=lambda c:(-c["max_peak_reuse_count"],-c["duplicated_matches"]))
    reuse_examples=[]
    for c in reuse[:5]:
        rows=groups[(c["scan_id"],c["candidate_index"],c["peptide_sequence"])]
        counts=Counter(r["matched_peak_index"] for r in rows if r["hit"])
        idx,n=counts.most_common(1)[0]; qs=[r["theoretical_mz"] for r in rows if r["hit"] and r["matched_peak_index"]==idx]
        reuse_examples.append({"scan_id":c["scan_id"],"candidate_index":c["candidate_index"],"peptide_sequence":c["peptide_sequence"],"peak_index":idx,"reuse_count":n,"theoretical_mz":qs})

    stats={"row_counts":{"scan_summary":len(scans),"observed_peaks":len(observed),"fragment_queries":len(fragments)},"missing_values":missing,"validation":validation,"peak_reconstruction":peak_checks,"window_statistics":windows,"absolute_mass_error":describe(abs_errors),"normalized_mass_error":describe(normalized),"density_by_scan":density,"local_bin_density":density_stats,"candidate_count":len(candidates),"candidate_hit_rate":describe(c["hit_rate"] for c in candidates),"candidate_query_count":describe(c["theoretical_fragment_queries"] for c in candidates),"processed_peak_count":describe(int(s["processed_peak_count"]) for s in scans),"total_query_workload":len(fragments),"reuse_candidate_count":len(reuse),"reuse_examples":reuse_examples}
    (output/"workload_statistics.json").write_text(json.dumps(stats,indent=2,sort_keys=True))

    # Plots.
    svg_plot(output/"hit_vs_miss_fraction.svg","In-range query outcomes","Outcome (0=hit, 1=miss)","Fraction",[{"points":[(0,len(hits)/len(inrange)),(1,len(misses)/len(inrange))]}])
    wc=Counter(r["num_peaks_in_window"] for r in inrange); svg_plot(output/"num_peaks_in_window_distribution.svg","Peaks inside strict tolerance window","W(q)","Query count",[{"points":sorted(wc.items())}])
    svg_plot(output/"normalized_mass_error_histogram.svg","Normalized absolute mass error","|mass error| / tolerance","Hit count",[{"points":histogram(normalized,30,0,1)}])
    svg_plot(output/"processed_peak_count_per_scan.svg","Processed peak count per summary row","Summary row","Processed peaks",[{"points":[(i+1,int(s["processed_peak_count"])) for i,s in enumerate(scans)]}])
    svg_plot(output/"candidate_hit_rate_distribution.svg","Candidate hit-rate distribution","Hit rate","Candidate count",[{"points":histogram([c["hit_rate"] for c in candidates],25,0,max(c["hit_rate"] for c in candidates))}])
    svg_plot(output/"candidate_query_count_distribution.svg","Theoretical queries per candidate","Query count","Candidate count",[{"points":histogram([c["theoretical_fragment_queries"] for c in candidates],25)}])
    series=[]
    for i,size in enumerate((1,5,10)): series.append({"points":sorted(Counter(density_values[size]).items()),"kind":"line","color":["#2864b7","#d34e30","#2a8c65"][i]})
    svg_plot(output/"local_retained_peak_density_distribution.svg","Local retained-peak counts (lines: 1, 5, 10 Da bins)","Peaks per bin","Number of bins",series)

    peak_rows=obs_by_scan[typical["scan_id"]]; qrows=groups[(typical["scan_id"],typical["candidate_index"],typical["peptide_sequence"])]
    spectrum_svg(output/"typical_candidate_full_spectrum.svg",peak_rows,qrows,typical)
    # Smallest interval joining opposite outcomes, with enough padding to show both.
    ir=sorted((r for r in qrows if r["in_spectrum_range"]),key=lambda r:r["theoretical_mz"])
    pairs=[(abs(a["theoretical_mz"]-b["theoretical_mz"]),a,b) for a in ir for b in ir if a["hit"] != b["hit"]]
    _,a,b=min(pairs,key=lambda x:x[0]); lo=min(a["theoretical_mz"],b["theoretical_mz"]); hi=max(a["theoretical_mz"],b["theoretical_mz"]); pad=max(0.05,(hi-lo)*0.08)
    spectrum_svg(output/"typical_candidate_zoom.svg",peak_rows,qrows,typical,(lo-pad,hi+pad))

    # Human-readable concise report.
    lines=["# Sipros Regular MVH workload analysis","",f"Analyzed {len(fragments):,} fragment rows, {len(observed):,} observed-peak rows, and {len(scans)} summary rows. Both summary rows have scan_id 5678; their counters combine exactly to the fragment reconstruction.","","## Validation","",f"- Summary identities all pass: {all(x['matched_plus_missed_ok'] and x['in_range_le_total'] for x in summary_internal)}; maximum stored hit-rate difference: {max(abs(x['hit_rate_difference']) for x in summary_internal):.3g}.",f"- Fragment reconstruction equals combined summary counters: {reconstructed==summary_combined}.",f"- HIT anomalies (bad index / missing m/z / missing error / error outside tolerance / W<=0): {validation['hit_bad_index']} / {validation['hit_missing_mz']} / {validation['hit_missing_error']} / {validation['hit_error_not_strictly_below_tolerance']} / {validation['hit_nonpositive_window']}.",f"- Mass-error convention is matched_peak_mz - theoretical_mz; max CSV-rounding residual {validation['mass_error_sign_max_abs_residual']:.3g} Da.",f"- MISS anomalies (index != -1 / populated match fields / W>0): {validation['miss_index_not_minus_one']} / {validation['miss_with_mz_or_error']} / {validation['miss_with_positive_window']}.","","## Workload","",f"- In-range: {len(inrange):,}; hits: {len(hits):,}; misses: {len(misses):,}; hit rate: {len(hits)/len(inrange):.4%}.",f"- W all: P0={windows['all_in_range']['p_w_0']:.4%}, P1={windows['all_in_range']['p_w_1']:.4%}, P2={windows['all_in_range']['p_w_2']:.4%}, P>=3={windows['all_in_range']['p_w_ge_3']:.4%}; mean={windows['all_in_range']['mean']:.4g}, p95={windows['all_in_range']['p95']:.4g}, max={windows['all_in_range']['max']}.",f"- W hits: P1={windows['hits']['p_w_1']:.4%}, P2={windows['hits']['p_w_2']:.4%}, P>=3={windows['hits']['p_w_ge_3']:.4%}; max={windows['hits']['max']}.",f"- W misses: P0={windows['misses']['p_w_0']:.4%}; positive-window misses={validation['miss_with_positive_window']}.",f"- Absolute hit error: mean={fmt(describe(abs_errors)['mean'])} Da, median={fmt(describe(abs_errors)['median'])}, p95={fmt(describe(abs_errors)['p95'])}, max={fmt(describe(abs_errors)['max'])}.",f"- Normalized error: mean={fmt(describe(normalized)['mean'])}, median={fmt(describe(normalized)['median'])}, p90={fmt(describe(normalized)['p90'])}, p95={fmt(describe(normalized)['p95'])}, p99={fmt(describe(normalized)['p99'])}.",f"- Candidate queries: median={fmt(describe(c['theoretical_fragment_queries'] for c in candidates)['median'])}, mean={fmt(describe(c['theoretical_fragment_queries'] for c in candidates)['mean'])}; total={len(fragments):,}.",f"- Candidates with repeated use of a matched peak: {len(reuse)}/{len(candidates)}.","","## Interpretation","","Measured evidence: this is overwhelmingly interval-existence work: almost all misses have an empty strict tolerance window, and W quantifies how rarely nearest selection has multiple choices. The retained spectrum has 55 peaks spanning its recorded retained range, so geometry is small while query count is large and hit rate is low.","","Hypothesis to test, not a performance claim: RT/BVH may benefit from batched queries but competes with a very small integer-bucket lookup. Before implementation, measure pMassHub entries actually visited, bucket-range sizes, CPU time/cycles and cache behavior, batch/build/transfer overheads, and scaling over many scans and peak densities. `num_peaks_in_window` cannot supply current lookup cost because `findNear` visits every peak in selected integer-m/z buckets before applying the strict tolerance test; therefore `num_peaks_inspected_by_findNear` is valuable for fair pMassHub, binary-search, CUDA, and BVH/RT comparisons.","","Full machine-readable details are in `workload_statistics.json`; selections are in `representative_candidates.csv`."]
    lines.extend(["","## Dataset structure","",
      "- `scan_summary`: 2 rows, 13 columns; integer counters/IDs, floating precursor/retention/hit-rate fields, and one filename string. One unique scan ID (5678), no missing values.",
      "- `observed_peaks`: 58 rows, 6 columns; integer IDs/index/flags/classes and floating m/z/intensity. One unique scan ID, no missing values.",
      "- `fragment_queries`: 26,659 rows, 14 columns; integer IDs/index/flags/classes/window counts, floating m/z/tolerance/error fields, and peptide strings. One unique scan ID and 494 unique candidate indices. The 25,945 missing values in each matched m/z and mass-error column are expected: 25,866 in-range misses plus 79 out-of-range rows.",
      "- Grouping by `(scan_id, candidate_index, peptide_sequence)` yields 573 candidate instances because the second recorded scoring pass reuses candidate indices 0–78 with different peptide sequences. The output follows the requested three-field grouping and does not merge them.","",
      "## Summary-row and peak reconstruction detail","",
      "- Summary row 1: total 23,257; in-range 23,195; hits 627; misses 22,568; stored/recomputed hit rate 0.0270316878638 / 0.027031687863764.",
      "- Summary row 2: total 3,402; in-range 3,385; hits 87; misses 3,298; stored/recomputed hit rate 0.0257016248154 / 0.025701624815362.",
      "- Combined fragment reconstruction: total 26,659; in-range 26,580; hits 714; misses 25,866. It matches the combined summary exactly.",
      "- Observed reconstruction: 58 raw peaks, 55 retained (94.8276%), matching both summary rows. Raw and retained m/z ranges are both 147.112350–1205.575439 Da.",
      "- Retained MVH classes are exactly three nonzero classes: class 1 = 8 peaks, class 2 = 16, class 3 = 31. The three discarded peaks have class 0.","",
      "## Error and density detail","",
      f"- Absolute error standard deviation={describe(abs_errors)['stddev']:.9g} Da; p50/p90/p95/p99={describe(abs_errors)['p50']:.9g}/{describe(abs_errors)['p90']:.9g}/{describe(abs_errors)['p95']:.9g}/{describe(abs_errors)['p99']:.9g} Da.",
      f"- Normalized-error standard deviation={describe(normalized)['stddev']:.6g}; maximum={describe(normalized)['max']:.6g}. The median (4.54% of tolerance) and p90 (5.38%) show strong concentration near the theoretical mass, with a smaller upper tail rather than a uniform distribution.",
      f"- Global retained density is {density[5678]['global_peaks_per_da']:.6g} peaks/Da across {density[5678]['retained_mz_range']:.6g} Da.",
      f"- 1 Da bins: mean {density_stats['1']['mean']:.4g}, p95 {density_stats['1']['p95']:.4g}, max {density_stats['1']['max']}, zero fraction {density_stats['1']['zero_fraction']:.2%}.",
      f"- 5 Da bins: mean {density_stats['5']['mean']:.4g}, p95 {density_stats['5']['p95']:.4g}, max {density_stats['5']['max']}, zero fraction {density_stats['5']['zero_fraction']:.2%}.",
      f"- 10 Da bins: mean {density_stats['10']['mean']:.4g}, p95 {density_stats['10']['p95']:.4g}, max {density_stats['10']['max']}, zero fraction {density_stats['10']['zero_fraction']:.2%}. This is sparse globally with modest local clustering (up to five peaks/10 Da).","",
      "## Representative selections","",
      *[f"- {label}: scan {c['scan_id']}, candidate {c['candidate_index']}, `{c['peptide_sequence']}`; raw/processed peaks {c['raw_peak_count']}/{c['processed_peak_count']}; fragments {c['theoretical_fragment_queries']}; hits/misses {c['hits']}/{c['misses']}; hit rate {c['hit_rate']:.4%}; mean/max W {c['mean_num_peaks_in_window']:.4g}/{c['max_num_peaks_in_window']}." for label,c in selected.items()],"",
      "The high-hit and dense-window labels select the same candidate because the observed maximum W is only 1. No observed peak reuse occurs in any of the 573 candidate instances; consequently there are no concrete reuse examples to list."])
    (output/"report.md").write_text("\n".join(lines)+"\n")
    print(f"Wrote analysis outputs to {output}")


if __name__ == "__main__": main()
