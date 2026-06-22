import argparse
import html
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return rows


def parse_json_text(value):
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def asset_url(path_text: str, project_root: Path) -> str:
    path = Path(path_text)
    resolved = path if path.is_absolute() else path.resolve()

    try:
        relative = resolved.relative_to(project_root)
        return "/" + relative.as_posix()
    except ValueError:
        return resolved.as_uri()


def e(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def badge_class(risk_state: str | None) -> str:
    if risk_state == "danger":
        return "danger"
    if risk_state == "unsafe":
        return "unsafe"
    if risk_state == "normal":
        return "normal"
    return "unknown"


def bool_value(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None

    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def render_image(path_text: str, label: str, project_root: Path) -> str:
    uri = asset_url(path_text, project_root)
    return f"""
        <a class="frame" href="{e(uri)}" target="_blank" title="{e(path_text)}">
          <img src="{e(uri)}" loading="lazy" alt="{e(label)}">
          <span>{e(label)}</span>
        </a>
    """


def render_row(row: dict, project_root: Path) -> str:
    frame_indices = parse_json_text(row.get("frame_indices"))
    frame_times_sec = parse_json_text(row.get("frame_times_sec"))
    stage1_actions = parse_json_text(row.get("stage1_actions"))

    if not isinstance(frame_indices, list):
        frame_indices = ["?", "?", "?"]
    if not isinstance(frame_times_sec, list):
        frame_times_sec = ["?", "?", "?"]
    if not isinstance(stage1_actions, list):
        stage1_actions = ["?", "?", "?"]

    pred = row.get("pred_risk_state")
    gt = row.get("sequence_gt") or row.get("gt_risk_state")
    match = pred == gt if pred and gt in {"normal", "unsafe", "danger"} else None
    match_text = "match" if match is True else "miss" if match is False else "unknown"
    eval_text = "eval" if bool_value(row.get("use_for_eval")) is not False else "no-eval"
    folder = row.get("folder_binary_label") or row.get("folder_coarse_label")
    search_text = " ".join([
        str(row.get("request_id", "")),
        str(row.get("group", "")),
        str(row.get("video_id", "")),
        str(folder or ""),
        str(pred or ""),
        str(gt or ""),
        match_text,
        " ".join(str(x) for x in stage1_actions),
    ]).lower()

    image_paths = [
        row.get(f"image_{idx + 1}", "")
        for idx in range(3)
        if row.get(f"image_{idx + 1}", "")
    ]

    frames = []
    for idx, _image_path in enumerate(image_paths):
        frame = frame_indices[idx] if idx < len(frame_indices) else "?"
        time_sec = frame_times_sec[idx] if idx < len(frame_times_sec) else "?"
        label = f"F{idx + 1} frame={frame} time={time_sec}s"
        frames.append(render_image(row.get(f"image_{idx + 1}", ""), label, project_root))

    actions = "\n".join(
        f"F{idx + 1}: {action}"
        for idx, action in enumerate(stage1_actions[:3])
    )

    return f"""
      <article class="item" data-risk="{e(pred)}" data-gt="{e(gt)}" data-match="{e(match_text)}" data-folder="{e(folder)}" data-search="{e(search_text)}">
        <header>
          <div>
            <h2>{e(row.get("request_id"))}</h2>
            <p>{e(row.get("group"))} / {e(row.get("video_id"))}</p>
          </div>
          <div class="badges">
            <span class="badge {badge_class(pred)}">pred {e(pred)}</span>
            <span class="badge {badge_class(gt)}">gt {e(gt)}</span>
            <span class="badge {e(match_text)}">{e(match_text)}</span>
            <span class="badge eval">{e(eval_text)}</span>
            <span class="badge folder">{e(folder)}</span>
          </div>
        </header>
        <div class="frames">
          {''.join(frames)}
        </div>
        <div class="meta">
          <pre>{e(actions)}</pre>
          <dl>
            <dt>frames</dt><dd>{e(frame_indices)}</dd>
            <dt>times</dt><dd>{e(frame_times_sec)}</dd>
            <dt>latency</dt><dd>{e(row.get("latency_sec"))}s</dd>
            <dt>gt</dt><dd>{e(gt)}</dd>
            <dt>pred</dt><dd>{e(pred)}</dd>
            <dt>match</dt><dd>{e(match_text)}</dd>
            <dt>json/schema</dt><dd>{e(row.get("json_success"))} / {e(row.get("schema_success"))}</dd>
          </dl>
        </div>
      </article>
    """


def build_html(rows: list[dict], title: str, project_root: Path) -> str:
    counts = {}
    gt_counts = {}
    match_counts = {}
    for row in rows:
        pred = row.get("pred_risk_state") or "unknown"
        gt = row.get("sequence_gt") or row.get("gt_risk_state") or "unknown"
        match = "match" if pred == gt else "miss"
        counts[pred] = counts.get(pred, 0) + 1
        gt_counts[gt] = gt_counts.get(gt, 0) + 1
        match_counts[match] = match_counts.get(match, 0) + 1

    count_text = " / ".join(
        f"{key}: {value}"
        for key, value in sorted(counts.items())
    )
    gt_count_text = " / ".join(
        f"{key}: {value}"
        for key, value in sorted(gt_counts.items())
    )
    match_count_text = " / ".join(
        f"{key}: {value}"
        for key, value in sorted(match_counts.items())
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{e(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      background: #f6f7f9;
      color: #171a1f;
    }}
    body {{
      margin: 0;
    }}
    .top {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: #ffffff;
      border-bottom: 1px solid #d8dde6;
      padding: 12px 16px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 18px;
      font-weight: 700;
    }}
    .controls {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }}
    input, select {{
      height: 32px;
      border: 1px solid #b8c0cc;
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      font-size: 14px;
    }}
    input {{
      min-width: 280px;
      flex: 1;
    }}
    .count {{
      color: #566070;
      font-size: 13px;
    }}
    main {{
      padding: 12px 16px 32px;
      display: grid;
      gap: 12px;
    }}
    .item {{
      background: #ffffff;
      border: 1px solid #d8dde6;
      border-radius: 8px;
      padding: 12px;
    }}
    .item[hidden] {{
      display: none;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      margin-bottom: 10px;
    }}
    h2 {{
      margin: 0 0 4px;
      font-size: 15px;
      line-height: 1.25;
      word-break: break-all;
    }}
    p {{
      margin: 0;
      color: #566070;
      font-size: 13px;
    }}
    .badges {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid transparent;
      white-space: nowrap;
    }}
    .danger {{ background: #ffe3e3; color: #9b111e; border-color: #ffb8b8; }}
    .unsafe {{ background: #fff0d6; color: #8a4a00; border-color: #ffd28a; }}
    .normal {{ background: #e5f6ea; color: #176b34; border-color: #b7e1c2; }}
    .unknown {{ background: #edf0f4; color: #566070; border-color: #d8dde6; }}
    .folder {{ background: #eef2ff; color: #31406f; border-color: #cfd8ff; }}
    .match {{ background: #e5f6ea; color: #176b34; border-color: #b7e1c2; }}
    .miss {{ background: #ffe3e3; color: #9b111e; border-color: #ffb8b8; }}
    .eval {{ background: #f4eefc; color: #55307b; border-color: #ddc8f6; }}
    .frames {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }}
    .frame {{
      display: grid;
      gap: 4px;
      text-decoration: none;
      color: #171a1f;
    }}
    .frame img {{
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #101217;
      border: 1px solid #d8dde6;
      border-radius: 6px;
    }}
    .frame span {{
      color: #566070;
      font-size: 12px;
    }}
    .meta {{
      margin-top: 10px;
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(260px, 1fr);
      gap: 10px;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      font-size: 13px;
      line-height: 1.4;
      background: #f6f7f9;
      border: 1px solid #e1e5ec;
      border-radius: 6px;
      padding: 8px;
    }}
    dl {{
      margin: 0;
      display: grid;
      grid-template-columns: 86px 1fr;
      gap: 4px 8px;
      font-size: 13px;
    }}
    dt {{
      color: #566070;
    }}
    dd {{
      margin: 0;
      word-break: break-word;
    }}
    @media (max-width: 760px) {{
      .frames, .meta {{
        grid-template-columns: 1fr;
      }}
      input {{
        min-width: 180px;
      }}
    }}
  </style>
</head>
<body>
  <section class="top">
    <h1>{e(title)}</h1>
    <div class="controls">
      <input id="q" type="search" placeholder="Search request, group, video, action">
      <select id="risk">
        <option value="">all pred</option>
        <option value="danger">danger</option>
        <option value="unsafe">unsafe</option>
        <option value="normal">normal</option>
      </select>
      <select id="gt">
        <option value="">all gt</option>
        <option value="danger">danger</option>
        <option value="unsafe">unsafe</option>
        <option value="normal">normal</option>
      </select>
      <select id="match">
        <option value="">all match</option>
        <option value="miss">miss</option>
        <option value="match">match</option>
      </select>
      <select id="folder">
        <option value="">all folder labels</option>
        <option value="intrusion">intrusion</option>
        <option value="normal">normal</option>
      </select>
      <span class="count" id="count"></span>
    </div>
    <div class="count">{len(rows)} rows / pred: {e(count_text)} / gt: {e(gt_count_text)} / {e(match_count_text)}</div>
  </section>
  <main id="items">
    {''.join(render_row(row, project_root) for row in rows)}
  </main>
  <script>
    const q = document.getElementById('q');
    const risk = document.getElementById('risk');
    const gt = document.getElementById('gt');
    const match = document.getElementById('match');
    const folder = document.getElementById('folder');
    const count = document.getElementById('count');
    const items = Array.from(document.querySelectorAll('.item'));

    function applyFilters() {{
      const query = q.value.trim().toLowerCase();
      const riskValue = risk.value;
      const gtValue = gt.value;
      const matchValue = match.value;
      const folderValue = folder.value;
      let visible = 0;

      for (const item of items) {{
        const okQuery = !query || item.dataset.search.includes(query);
        const okRisk = !riskValue || item.dataset.risk === riskValue;
        const okGt = !gtValue || item.dataset.gt === gtValue;
        const okMatch = !matchValue || item.dataset.match === matchValue;
        const okFolder = !folderValue || item.dataset.folder === folderValue;
        const show = okQuery && okRisk && okGt && okMatch && okFolder;
        item.hidden = !show;
        if (show) visible += 1;
      }}

      count.textContent = `${{visible}} visible`;
    }}

    q.addEventListener('input', applyFilters);
    risk.addEventListener('change', applyFilters);
    gt.addEventListener('change', applyFilters);
    match.addEventListener('change', applyFilters);
    folder.addEventListener('change', applyFilters);
    applyFilters();
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--project-root", default=".")

    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    project_root = Path(args.project_root).resolve()
    raw_path = result_dir / "raw_results.jsonl"

    if not raw_path.exists():
        raise FileNotFoundError(f"raw_results.jsonl not found: {raw_path}")

    output_path = Path(args.output).resolve() if args.output else result_dir / "viewer.html"
    rows = load_jsonl(raw_path)
    html_text = build_html(rows, title=result_dir.name, project_root=project_root)

    output_path.write_text(html_text, encoding="utf-8")
    print("Saved viewer:", output_path)


if __name__ == "__main__":
    main()


# python3 benchmark2/scripts/build_result_viewer.py \
#   --result-dir benchmark2/results/qwen35_2b_v2_gt_all \
#   --project-root /home/capstone2/zroact-stage2


# cd /home/capstone2/zroact-stage2
# python3 -m http.server 8000 --bind 127.0.0.1
