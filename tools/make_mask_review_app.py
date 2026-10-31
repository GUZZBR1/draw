#!/usr/bin/env python3
"""Create a private, offline, full-resolution review app for mask proposals.

The generated app supports blinded image-level observations, overlay review,
and optional categorical brush corrections saved as separate grayscale PNGs.
It records reviewer decisions but does not decide consensus or promote labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "captures/triage/pseudo_training_candidate/samples.csv"
DEFAULT_OUTPUT = ROOT / "captures/triage/mask_review_app"
ALLOWED_MASK_VALUES = {0, 1, 2, 3, 255}
PALETTE_BGR = {
    0: (45, 45, 45),
    1: (75, 190, 95),
    2: (240, 95, 35),
    3: (25, 90, 245),
    255: (205, 0, 205),
}
OVERLAY_ALPHA = 145


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(value: str, *, base: Path) -> Path:
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"input path escapes repository: {value}")
    return path


def safe_relative(path: Path, base: Path) -> str:
    return quote(Path(os.path.relpath(path, base)).as_posix(), safe="/._-()")


def read_rows(path: Path, session_id: str | None, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "session_id", "frame_index", "source_path", "source_sha256",
            "mask_path", "mask_sha256", "width", "height",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(sorted(missing))}")
        rows = [row for row in reader if not session_id or row["session_id"] == session_id]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("no frames matched the requested input/session/limit")
    ids = [(row["session_id"], int(row["frame_index"])) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("input contains duplicate session/frame IDs")
    return rows


def prepare_records(
    rows: list[dict[str, str]], input_path: Path, output_dir: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_blind_ids: set[str] = set()
    media_dir = output_dir / "media"
    labels_dir = output_dir / "labels"
    (output_dir / "corrections").mkdir(parents=True)
    media_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for row in rows:
        source = resolve_repo_path(row["source_path"], base=ROOT)
        mask_path = resolve_repo_path(row["mask_path"], base=input_path.parent)
        if not source.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"missing source image or mask for {row['session_id']} frame {row['frame_index']}")

        actual_source_hash = sha256(source)
        actual_mask_hash = sha256(mask_path)
        if actual_source_hash != row["source_sha256"]:
            raise ValueError(f"source SHA-256 mismatch: {source}")
        if actual_mask_hash != row["mask_sha256"]:
            raise ValueError(f"mask SHA-256 mismatch: {mask_path}")

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise ValueError(f"cannot decode source/mask for {row['session_id']} frame {row['frame_index']}")
        if mask.ndim != 2:
            raise ValueError(f"mask must be single-channel: {mask_path}")
        height, width = image.shape[:2]
        if mask.shape != (height, width):
            raise ValueError(f"source/mask dimensions differ: {source} and {mask_path}")
        if (width, height) != (int(row["width"]), int(row["height"])):
            raise ValueError(f"manifest dimensions differ from decoded image: {source}")
        invalid = sorted(set(int(value) for value in np.unique(mask)) - ALLOWED_MASK_VALUES)
        if invalid:
            raise ValueError(f"mask has unsupported pixel values {invalid}: {mask_path}")

        blind_id = hashlib.sha256(
            f"{row['session_id']}:{row['frame_index']}:{actual_source_hash}:{actual_mask_hash}".encode()
        ).hexdigest()[:14]
        if blind_id in seen_blind_ids:
            raise ValueError(f"blind-ID collision: {blind_id}")
        seen_blind_ids.add(blind_id)
        # Copy the exact source bytes and the lossless mask into this private
        # batch. The browser can then read both from localhost without exposing
        # original filenames to reviewers or tainting the correction canvas.
        source_copy = media_dir / f"{blind_id}{source.suffix.lower()}"
        mask_copy = labels_dir / f"{blind_id}.png"
        shutil.copyfile(source, source_copy)
        shutil.copyfile(mask_path, mask_copy)
        if sha256(source_copy) != actual_source_hash or sha256(mask_copy) != actual_mask_hash:
            raise ValueError(f"review copy integrity check failed for {blind_id}")

        records.append({
            "blind_id": blind_id,
            "session_id": row["session_id"],
            "frame_index": int(row["frame_index"]),
            "source_path": str(source.relative_to(ROOT)),
            "source_sha256": actual_source_hash,
            "mask_path": str(mask_path.relative_to(ROOT)),
            "mask_sha256": actual_mask_hash,
            "review_copy_source": str(source_copy.relative_to(ROOT)),
            "review_copy_mask": str(mask_copy.relative_to(ROOT)),
            "width": width,
            "height": height,
            "source_url": safe_relative(source_copy, output_dir),
            "mask_url": safe_relative(mask_copy, output_dir),
            "correction_dir": str((output_dir / "corrections").relative_to(ROOT)),
        })
    return records


def make_html(records: list[dict[str, object]], created_utc: str) -> str:
    # The manifest mapping is necessary for provenance and is downloadable
    # alongside reviews, but the rendered page never displays paths or tags.
    embedded = json.dumps(records, separators=(",", ":")).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Draw mask review (offline)</title>
<style>
:root{{color-scheme:dark;font:15px system-ui,sans-serif;background:#111827;color:#e5e7eb}}
*{{box-sizing:border-box}} body{{margin:0}} header{{padding:16px 20px;background:#172033;position:sticky;top:0;z-index:5;border-bottom:1px solid #344055}}
h1{{font-size:19px;margin:0 0 8px}} .notice{{color:#fcd34d;font-size:13px;max-width:1100px}}
main{{max-width:1400px;margin:0 auto;padding:16px}} .bar,.row{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
.bar{{padding:8px 0 14px}} button,select,input,textarea{{font:inherit;color:inherit;background:#202b3e;border:1px solid #536178;border-radius:6px;padding:8px}}
button{{cursor:pointer}} button:hover{{background:#33435b}} button:disabled{{opacity:.45;cursor:not-allowed}}
label{{display:inline-flex;gap:6px;align-items:center}} #reviewer{{width:210px}} #stage{{background:#182235;padding:14px;border-radius:10px}}
#images{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;overflow:auto;max-height:68vh}}
.pane{{min-width:0}} .pane-title{{font-size:13px;color:#cbd5e1;margin:0 0 6px}} .viewport{{overflow:auto;max-height:62vh;background:#05070b;border:1px solid #374151}}
.viewport img{{display:block;max-width:none;transform-origin:top left}} #editor{{display:block;max-width:none;transform-origin:top left;touch-action:none;cursor:crosshair}} .legend{{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0;font-size:12px}}
.swatch{{display:inline-block;width:12px;height:12px;border:1px solid #aaa;margin-right:4px;vertical-align:-2px}}
.formgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:12px 0}}
fieldset{{border:1px solid #46536a;border-radius:8px;padding:10px}} legend{{color:#cbd5e1}} textarea{{width:100%;min-height:72px}}
#status{{color:#a7f3d0;min-height:1.5em}} .small{{font-size:12px;color:#9ca3af}} .hidden{{display:none!important}}
@media(max-width:900px){{#images{{grid-template-columns:1fr}}}}
</style></head>
<body><header><h1>Revisão local de máscaras — avaliações independentes</h1>
<div class="notice">Ferramenta offline: nada é enviado pela rede. A máscara é uma proposta; registrar uma decisão não a torna verdade-terreno nem a aprova automaticamente para treino. Para independência, use um ID de revisor próprio e não consulte a avaliação de outra pessoa.</div>
</header><main>
<section class="bar"><label>Reviewer ID <input id="reviewer" autocomplete="off" placeholder="ex.: vision-reviewer-A"></label>
<button id="start">Start / resume</button><span id="progress" class="small"></span><span id="status" role="status"></span></section>
<section id="stage" class="hidden">
<div class="bar"><strong id="frame-label"></strong><button id="prev">← Previous</button><button id="next">Next →</button>
<label>Zoom <input id="zoom" type="range" min="50" max="250" value="100"><span id="zoom-label">100%</span></label>
<button id="reveal">Lock blind observations and reveal mask</button></div>
<div class="formgrid"><fieldset><legend>Phase 1 — image only, before mask reveal</legend>
<div id="tags" class="row"></div><label class="small">Image confidence <select id="image-confidence"><option value="">Choose</option><option>low</option><option>medium</option><option>high</option></select></label>
<label>Source assessment <select id="source-outcome"><option value="">Choose</option><option value="accept">accept (image assessable)</option><option value="edit">edit/annotate needed</option><option value="reject">reject source</option><option value="uncertain">uncertain</option></select></label>
<textarea id="image-notes" placeholder="Visible-content notes; assess the source image before revealing the proposal"></textarea></fieldset>
<fieldset id="decision-box" class="hidden"><legend>Phase 2 — compare proposal against source</legend>
<label>Review phase <select id="review-phase"><option value="overlay_review">independent overlay review</option><option value="adjudication">third-reviewer adjudication</option></select></label>
<label>Overlay decision <select id="outcome"><option value="">Choose</option><option value="accept">accept</option><option value="edit">edit required</option><option value="reject">reject</option><option value="uncertain">uncertain</option></select></label>
<label>Overlay confidence <select id="overlay-confidence"><option value="">Choose</option><option>low</option><option>medium</option><option>high</option></select></label>
<label><input type="checkbox" id="edit-mode"> Brush-edit proposal</label>
<label>Paint class <select id="paint-class"><option value="0">0 background</option><option value="1">1 paper</option><option value="2">2 hand</option><option value="3">3 user ink</option><option value="255">255 unknown</option></select></label>
<label>Brush size (source pixels) <input id="brush-size" type="number" min="1" max="80" value="8"></label><button id="undo-stroke" type="button">Undo last stroke</button>
<textarea id="overlay-notes" placeholder="Describe mismatched regions/classes or why uncertain. Brush edits export a separate corrected grayscale PNG; the source mask stays unchanged."></textarea>
<button id="save">Save this review decision</button></fieldset></div>
<div id="images"><div class="pane"><p class="pane-title">Original image (native resolution; scroll to pan)</p><div class="viewport"><img id="source" alt="Source frame"></div></div>
<div class="pane hidden" id="overlay-pane"><p class="pane-title">Original image + proposal overlay (native pixels; edit mode is optional)</p><div class="viewport" style="position:relative"><img id="source-overlay-base" alt="Source frame behind proposal"><canvas id="editor" aria-label="Mask overlay canvas"></canvas></div>
<label class="small">Overlay opacity <input id="opacity" type="range" min="0" max="100" value="62"></label></div></div>
<div class="legend"><span><i class="swatch" style="background:#2d2d2d"></i>background</span><span><i class="swatch" style="background:#5fbe4b"></i>paper</span><span><i class="swatch" style="background:#235ff0"></i>hand</span><span><i class="swatch" style="background:#f55a19"></i>user ink</span><span><i class="swatch" style="background:#cd00cd"></i>unknown (255)</span></div>
</section>
<section class="bar"><button id="export" disabled>Export append-only review CSV</button><span class="small">Records stay in this browser’s local storage until exported. Export files are reviewer-specific; merge/adjudicate them separately.</span></section>
<p class="small">Generated {html.escape(created_utc)} · {len(records)} frames · app v1. This workflow captures specialist review evidence; it does not establish human ground truth.</p>
</main><script>
const frames={embedded};
const tags=["paper_visible","hand_visible","user_ink_visible","printed_marks_visible","hand_over_ink","pen_visible","no_paper","blur_or_glare"];
const appKey="draw-mask-review-v1:"+{json.dumps(hashlib.sha256(embedded.encode()).hexdigest()[:16])}+":";
let reviewer="", index=0, records=[], blindDraft=null, revealed=false;
let labels=null, edited=false, undoStack=[];
const el=id=>document.getElementById(id);
function storageKey(){{return appKey+reviewer}}
function saveLocal(){{localStorage.setItem(storageKey(),JSON.stringify(records)); el("export").disabled=records.length===0; updateProgress()}}
function updateProgress(){{const done=new Set(records.filter(r=>r.phase==="overlay_review"||r.phase==="adjudication").map(r=>`${{r.session_id}}:${{r.frame_index}}`)).size;el("progress").textContent=reviewer?`${{done}} frames reviewed / ${{frames.length}}`:""}}
function setStatus(msg){{el("status").textContent=msg; setTimeout(()=>{{if(el("status").textContent===msg)el("status").textContent=""}},3500)}}
function clearTags(){{document.querySelectorAll("#tags input").forEach(x=>x.checked=false)}}
function getTags(){{return [...document.querySelectorAll("#tags input:checked")].map(x=>x.value)}}
function renderTags(){{el("tags").innerHTML=tags.map(t=>`<label><input type="checkbox" value="${{t}}">${{t.replaceAll("_"," ")}}</label>`).join("")}}
function render(){{
 const f=frames[index]; if(!f)return;
 el("frame-label").textContent=`Frame ${{index+1}} / ${{frames.length}} · blinded ID ${{f.blind_id}}`;
 el("source").src=f.source_url; el("source-overlay-base").src=f.source_url;
 el("source").style.width=`${{f.width}}px`; el("source-overlay-base").style.width=`${{f.width}}px`;
 el("source").style.height=`${{f.height}}px`; el("source-overlay-base").style.height=`${{f.height}}px`;
 el("overlay-pane").classList.toggle("hidden",!revealed); el("decision-box").classList.toggle("hidden",!revealed);
 el("reveal").classList.toggle("hidden",revealed);
 el("prev").disabled=index===0; el("next").disabled=index===frames.length-1;
 if(revealed)loadMask(f);
 if(blindDraft){{document.querySelectorAll("#tags input").forEach(x=>x.checked=blindDraft.visible_tags.includes(x.value));el("image-confidence").value=blindDraft.image_confidence;el("source-outcome").value=blindDraft.source_outcome;el("image-notes").value=blindDraft.image_notes}}
}}
function getDraft(){{return {{visible_tags:getTags(),image_confidence:el("image-confidence").value,source_outcome:el("source-outcome").value,image_notes:el("image-notes").value}}}}
const colors={{0:[50,50,50],1:[75,190,95],2:[240,95,35],3:[25,90,245],255:[205,0,205]}};
function paintOverlay(){{if(!labels)return;const f=frames[index],c=el("editor"),ctx=c.getContext("2d"),im=ctx.createImageData(f.width,f.height);for(let i=0;i<labels.length;i++){{const col=colors[labels[i]]||colors[255],p=i*4;im.data[p]=col[0];im.data[p+1]=col[1];im.data[p+2]=col[2];im.data[p+3]=labels[i]===0?35:150}}ctx.putImageData(im,0,0);c.style.width=`${{f.width}}px`;c.style.height=`${{f.height}}px`;c.style.position="absolute";c.style.left="0";c.style.top="0";c.style.opacity=Number(el("opacity").value)/100}}
function loadMask(f){{labels=null;edited=false;undoStack=[];const img=new Image();img.onload=()=>{{const c=document.createElement("canvas");c.width=f.width;c.height=f.height;const x=c.getContext("2d",{{willReadFrequently:true}});x.drawImage(img,0,0);const data=x.getImageData(0,0,f.width,f.height).data;labels=new Uint8Array(f.width*f.height);for(let i=0;i<labels.length;i++)labels[i]=data[i*4];el("editor").width=f.width;el("editor").height=f.height;paintOverlay()}};img.onerror=()=>setStatus("Could not load mask; ensure local server is running.");img.src=f.mask_url}}
function resize(){{const v=Number(el("zoom").value)/100; for(const id of ["source","source-overlay-base","editor"])el(id).style.transform=`scale(${{v}})`;el("zoom-label").textContent=el("zoom").value+"%";paintOverlay()}}
function pngChunk(type,data){{const b=new Uint8Array(data.length+12),v=new DataView(b.buffer);v.setUint32(0,data.length);for(let i=0;i<4;i++)b[4+i]=type.charCodeAt(i);b.set(data,8);let crc=0xffffffff;for(let i=4;i<8+data.length;i++){{crc^=b[i];for(let j=0;j<8;j++)crc=(crc>>>1)^((crc&1)?0xedb88320:0)}}v.setUint32(8+data.length,(crc^0xffffffff)>>>0);return b}}
async function encodeGrayPng(w,h,pixels){{const raw=new Uint8Array(h*(w+1));for(let y=0;y<h;y++)raw.set(pixels.subarray(y*w,(y+1)*w),y*(w+1)+1);const compressed=new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream("deflate"))).arrayBuffer());const ihdr=new Uint8Array(13),dv=new DataView(ihdr.buffer);dv.setUint32(0,w);dv.setUint32(4,h);ihdr[8]=8;ihdr[9]=0;const parts=[new Uint8Array([137,80,78,71,13,10,26,10]),pngChunk("IHDR",ihdr),pngChunk("IDAT",compressed),pngChunk("IEND",new Uint8Array())];const out=new Uint8Array(parts.reduce((n,p)=>n+p.length,0));let off=0;for(const p of parts){{out.set(p,off);off+=p.length}}return out}}
renderTags();
el("start").onclick=()=>{{reviewer=el("reviewer").value.trim();if(!reviewer){{setStatus("Enter a reviewer ID first.");return}}try{{records=JSON.parse(localStorage.getItem(storageKey())||"[]")}}catch{{records=[]}}el("stage").classList.remove("hidden");saveLocal();render();setStatus("Reviewer-local session started; other reviewers’ records are not loaded.")}};
el("prev").onclick=()=>{{if(index>0){{index--;blindDraft=null;revealed=false;render()}}}};
el("next").onclick=()=>{{if(index<frames.length-1){{index++;blindDraft=null;revealed=false;render()}}}};
el("zoom").oninput=resize;el("opacity").oninput=paintOverlay;
let drawing=false;
el("editor").onpointerdown=e=>{{if(!el("edit-mode").checked||!labels)return;drawing=true;el("editor").setPointerCapture(e.pointerId);undoStack.push(labels.slice());stroke(e)}};
el("editor").onpointermove=e=>{{if(drawing)stroke(e)}};el("editor").onpointerup=()=>{{drawing=false}};el("editor").onpointercancel=()=>{{drawing=false}};
function stroke(e){{const f=frames[index],r=el("editor").getBoundingClientRect(),scale=Number(el("zoom").value)/100,x=(e.clientX-r.left)/scale,y=(e.clientY-r.top)/scale,rad=Math.max(0,Math.floor(Number(el("brush-size").value)/2)),cx=Math.round(x),cy=Math.round(y),value=Number(el("paint-class").value);for(let yy=Math.max(0,cy-rad);yy<=Math.min(f.height-1,cy+rad);yy++)for(let xx=Math.max(0,cx-rad);xx<=Math.min(f.width-1,cx+rad);xx++)if((xx-cx)**2+(yy-cy)**2<=rad**2)labels[yy*f.width+xx]=value;edited=true;paintOverlay()}}
el("undo-stroke").onclick=()=>{{if(undoStack.length){{labels=undoStack.pop();edited=undoStack.length>0;paintOverlay()}}}};
el("reveal").onclick=()=>{{blindDraft=getDraft();if(!blindDraft.image_confidence||!blindDraft.source_outcome){{setStatus("Record source assessment and confidence before revealing the mask.");return}}revealed=true;render()}};
el("save").onclick=async()=>{{if(!reviewer||!revealed)return;const outcome=el("outcome").value;if(!outcome){{setStatus("Choose accept, edit, reject, or uncertain.");return}}if(edited&&outcome!=="edit"){{setStatus("Brush changes require outcome=edit.");return}}if(outcome==="edit"&&!edited){{setStatus("Use the brush to make a corrected mask before recording edit.");return}}const f=frames[index],reviewed=new Date().toISOString(),phase=el("review-phase").value;if(records.some(r=>r.reviewer_id===reviewer&&r.session_id===f.session_id&&r.frame_index===f.frame_index)){{setStatus("This reviewer already saved this frame; duplicate review records are blocked.");return}}let correctionPath="",correctionHash="";if(edited){{try{{const png=await encodeGrayPng(f.width,f.height,labels),digest=await crypto.subtle.digest("SHA-256",png),hex=[...new Uint8Array(digest)].map(x=>x.toString(16).padStart(2,"0")).join("");correctionPath=`${{f.correction_dir}}/${{f.blind_id}}_${{reviewer.replace(/[^a-zA-Z0-9_-]/g,"_")}}_${{Date.now()}}.png`;correctionHash=hex;const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([png],{{type:"image/png"}}));a.download=correctionPath.split("/").pop();a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}}catch(e){{setStatus("PNG correction export failed: "+e);return}}}}const common={{reviewer_id:reviewer,session_id:f.session_id,frame_index:f.frame_index,source_path:f.source_path,source_sha256:f.source_sha256,mask_path:f.mask_path,mask_sha256:f.mask_sha256,width:f.width,height:f.height,displayed_native_resolution:`${{f.width}}x${{f.height}}@${{el("zoom").value}}%`,corrected_mask_path:correctionPath,corrected_mask_sha256:correctionHash,reviewed_at_utc:reviewed}};records.push({{...common,phase:"blind_source",outcome:blindDraft.source_outcome,issues:blindDraft.visible_tags.join(";"),confidence:blindDraft.image_confidence,notes:blindDraft.image_notes}});records.push({{...common,phase,outcome,issues:blindDraft.visible_tags.join(";")+ (el("overlay-notes").value.trim()?";overlay_note":""),confidence:el("overlay-confidence").value,notes:el("overlay-notes").value.trim()}});saveLocal();setStatus(edited?"Review saved. Move downloaded correction PNG into the batch corrections folder; no mask was changed or promoted.":"Review recorded locally; it has not approved or changed the mask.");el("outcome").value="";el("overlay-confidence").value="";el("overlay-notes").value="";el("review-phase").value="overlay_review";el("source-outcome").value="";el("image-confidence").value="";el("image-notes").value="";clearTags();revealed=false;blindDraft=null;labels=null;edited=false;undoStack=[];render()}};
el("export").onclick=()=>{{if(!reviewer||!records.length)return;const columns=["reviewer_id","session_id","frame_index","phase","outcome","source_path","source_sha256","mask_path","mask_sha256","width","height","issues","confidence","notes","reviewed_at_utc","corrected_mask_path","corrected_mask_sha256","displayed_native_resolution"];const q=v=>`"${{String(v??"").replaceAll('"','""')}}"`;const body=[columns.map(q).join(","),...records.map(r=>columns.map(k=>q(r[k])).join(","))].join("\\r\\n")+"\\r\\n";const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([body],{{type:"text/csv;charset=utf-8"}}));a.download=`mask_reviews_${{reviewer.replace(/[^a-zA-Z0-9_-]/g,"_")}}_${{new Date().toISOString().replace(/[:.]/g,"-")}}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}};
</script></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="pseudo-label samples.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--session-id", help="optional exact session filter (useful for calibration batches)")
    parser.add_argument("--limit", type=int, help="optional first-N limit after session filtering")
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file() or not input_path.is_relative_to(ROOT):
        parser.error(f"--input must be a file inside this repository: {input_path}")
    if not output_dir.is_relative_to(ROOT / "captures/triage"):
        parser.error("--output-dir must stay inside captures/triage")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if output_dir.exists():
        parser.error(f"refusing to overwrite existing review output: {output_dir}")

    rows = read_rows(input_path, args.session_id, args.limit)
    output_dir.mkdir(parents=True)
    try:
        records = prepare_records(rows, input_path, output_dir)
        created_utc = datetime.now(timezone.utc).isoformat()
        manifest = {
            "schema_version": 1,
            "created_utc": created_utc,
            "review_status": "unreviewed_review_app; no labels are approved",
            "source_manifest": str(input_path.relative_to(ROOT)),
            "record_count": len(records),
            "app_file": "index.html",
            "review_policy": {
                "blind_first_phase": True,
                "mask_overlay_shown_after_blind_observations": True,
                "two_independent_reviewers_required": True,
                "third_adjudicator_for_disagreement_or_ink_print_occlusion": True,
                "consensus_is_not_human_ground_truth": True,
                "tool_promotes_masks": False,
            },
            "records": [{k: v for k, v in r.items() if not k.endswith("_url")} for r in records],
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (output_dir / "index.html").write_text(make_html(records, created_utc), encoding="utf-8")
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Draw mask review evidence record",
            "type": "object",
            "required": ["reviewer_id", "session_id", "frame_index", "phase", "outcome", "source_path", "source_sha256", "mask_path", "mask_sha256", "width", "height", "issues", "confidence", "notes", "reviewed_at_utc", "corrected_mask_path", "corrected_mask_sha256", "displayed_native_resolution"],
            "properties": {
                "schema_version": {"const": 1}, "reviewer_id": {"type": "string", "minLength": 1},
                "session_id": {"type": "string"}, "frame_index": {"type": "integer"},
                "phase": {"enum": ["blind_source", "overlay_review", "adjudication"]},
                "outcome": {"enum": ["accept", "edit", "reject", "uncertain"]},
                "source_path": {"type": "string"}, "mask_path": {"type": "string"},
                "blind_id": {"type": "string"}, "source_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "mask_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "width": {"type": "integer", "minimum": 1}, "height": {"type": "integer", "minimum": 1},
                "issues": {"type": "string"}, "confidence": {"type": "string"}, "notes": {"type": "string"},
                "corrected_mask_path": {"type": "string"}, "corrected_mask_sha256": {"type": "string"},
                "reviewed_at_utc": {"type": "string", "format": "date-time"},
                "displayed_native_resolution": {"type": "string"},
            },
        }
        (output_dir / "review_record.schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        readme = f'''# Offline mask-review batch

Generated at {created_utc} from `{input_path.relative_to(ROOT)}`; frames: {len(records)}.

## Open

Open `index.html` in a browser. If local-file restrictions prevent images from loading, run `python3 -m http.server 8765 --bind 127.0.0.1` from the repository root and open `http://127.0.0.1:8765/{output_dir.relative_to(ROOT).as_posix()}/`.

This app makes no network requests and does not modify source images or masks. Phase 1 records a reviewer’s assessment of the original image before the mask overlay is revealed. Phase 2 records accept/edit/reject/uncertain against that overlay. It supports zoom and scroll at native resolution.

## Review and export

Enter a distinct reviewer ID, complete both phases, then export CSV. Each reviewer must use their own ID and must not view other reviewers’ exports before completing their own independent pass. A third reviewer should select `third-reviewer adjudication` for all disagreements, ink-positive frames, printed-vs-user-mark ambiguity, and hand-over-ink cases. The app does not import the other reviewers’ files; the adjudicator's assignment and any conflict context must be given separately.

The CSV is **evidence of a review decision**, not an automatic approval. It has one `blind_source` row (raw image assessment) and one `overlay_review` or `adjudication` row per completed frame. Optional brush edits change an in-memory copy only. Saving a brush-edited frame downloads a separate one-channel grayscale PNG with the categorical values preserved (`0,1,2,3,255`) and records its suggested path and SHA-256. Move that downloaded file into this batch's `corrections/` folder before using it; the browser cannot write files into the batch directory. The original source and mask are never modified. `edit` without brush strokes is an edit request only. Do not call a frame consensus-reviewed until two independent overlay-review records and required adjudication are present. Do not call agent consensus human ground truth.

The `manifest.json` holds exact source/mask paths, SHA-256 hashes, IDs, and dimensions for audit. It deliberately omits semantic tag hints and previous review decisions from the rendered interface.
'''
        (output_dir / "README.md").write_text(readme, encoding="utf-8")
    except Exception:
        # Only remove the newly created, unreturned app directory; it contains
        # no user data and no source or mask is ever written there.
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    print(f"review app created: {output_dir}")
    print(f"frames: {len(records)}; source manifest: {input_path}")
    print("status: review evidence only; no mask was changed or promoted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
