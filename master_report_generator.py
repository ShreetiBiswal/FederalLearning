import argparse
import os
from datetime import datetime

from GraphPlotter.plot_final_results import plot_final_results
from GraphPlotter.server_global_plot import plot_metrics as plot_true_global_metrics
from GraphPlotter.average_local_accuracy_plot import plot_server_metrics
from GraphPlotter.plot_confusion_matrix import plot_confusion_matrix
from GraphPlotter.client_local_accuracy_plot import plot_client_metrics
from calculate_alpha import calculate_alpha

# =====================================================================
TARGET_ALGORITHMS = [
    "fedavg_nosmote",
    "wsm_ce_fedavg_nosmote",
    "scaffold_nosmote",
    "wsm_hm_class_weighted_nosmote"
]
# =====================================================================

HTML_STYLE = """
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  /* ── Screen styles ── */
  body {
    font-family: "Georgia", "Times New Roman", serif;
    background: #F5F4F0;
    color: #1a1a1a;
    font-size: 15px;
    line-height: 1.75;
  }

  .page-wrap {
    max-width: 960px;
    margin: 48px auto;
  }

  /* Each .page is a distinct white card on screen */
  .page {
    background: #FFFFFF;
    border: 1px solid #D8D5CC;
    /* Bottom padding is increased to give the absolute footer room to breathe */
    padding: 64px 72px 100px 72px; 
    margin-bottom: 32px;
    position: relative;
    min-height: 85vh; 
  }

  /* ── Enhanced Cover Layout ── */
  .page.cover {
    display: flex;
    flex-direction: column;
  }

  .cover-center {
    margin: auto 0;
    text-align: center;
    padding-bottom: 40px; /* Balance the vertical space */
  }

  .cover-inner {
    border-top: 3px solid #1a1a1a;
    border-bottom: 3px solid #1a1a1a;
    padding: 60px 0;
    margin: 0 auto;
    width: 100%;
  }

  .cover .kicker {
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 24px;
    font-weight: bold;
  }

  .cover h1 {
    font-size: 34px;
    font-weight: normal;
    line-height: 1.35;
    margin-bottom: 20px;
    color: #111;
  }

  .cover .subtitle {
    font-size: 16px;
    color: #444;
    font-style: italic;
    margin-bottom: 32px;
  }

  .badge-container {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 10px;
  }

  .badge {
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: #1a1a1a;
    color: #fff;
    padding: 6px 16px;
    border-radius: 3px;
    font-weight: bold;
  }
  .badge.alpha {
    background: #D85A30; 
    text-transform: none;}

  /* ── Absolute Bottom Anchored Footers ── */
  .cover-footer {
    border-top: 2px solid #1a1a1a;
    padding-top: 20px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    font-family: "Helvetica Neue", Arial, sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    /* Forces footer to the absolute bottom of the container */
    position: absolute;
    bottom: 64px;
    left: 72px;
    right: 72px;
  }
  .cover-footer > div:first-child {
    text-align: left;
  }
  .cover-footer strong {
    font-size: 12px;
    color: #111;
    display: block;
    margin-bottom: 4px;
  }
  .cover-footer span {
    font-size: 11px;
    color: #666;
  }

  .page-footer {
    border-top: 1px solid #E0DDD5;
    padding-top: 16px;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 11px;
    color: #AAA;
    display: flex;
    justify-content: space-between;
    /* Forces footer to the absolute bottom of the container */
    position: absolute;
    bottom: 64px;
    left: 72px;
    right: 72px;
  }

  /* ── TOC page ── */
  .toc-heading {
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 11px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 6px;
  }
  .toc-title {
    font-size: 20px;
    font-weight: normal;
    color: #111;
    border-bottom: 1px solid #D0CEC8;
    padding-bottom: 10px;
    margin-bottom: 32px;
  }
  .toc ol {
    padding-left: 0;
    list-style: none;
    counter-reset: toc-counter;
  }
  .toc li {
    counter-increment: toc-counter;
    display: flex;
    align-items: baseline;
    gap: 0;
    margin-bottom: 14px;
    font-size: 15px;
    color: #333;
  }
  .toc li::before {
    content: counter(toc-counter) ".";
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
    color: #999;
    min-width: 28px;
  }
  .toc li .toc-dots {
    flex: 1;
    border-bottom: 1px dotted #CCC;
    margin: 0 10px 4px;
  }
  .toc li .toc-page-num {
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
    color: #999;
  }
  .toc a { color: inherit; text-decoration: none; }
  .toc a:hover { text-decoration: underline; }

  /* ── Content pages ── */
  .section-number {
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 11px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 6px;
  }
  .section h2 {
    font-size: 24px;
    font-weight: normal;
    color: #111;
    border-bottom: 2px solid #111;
    padding-bottom: 10px;
    margin-bottom: 32px;
  }
  .section h3 {
    font-size: 18px;
    font-weight: normal;
    color: #333;
    border-bottom: 1px solid #D0CEC8;
    padding-bottom: 8px;
    margin-top: 40px;
    margin-bottom: 20px;
  }
  .section h4 {
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 11px;
    color: #777;
    font-weight: normal;
    margin: 28px 0 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  /* ── Figures ── */
  figure {
    margin: 24px 0 48px 0;
    text-align: center;
  }
  figure img {
    max-width: 100%;
    height: auto;
    border: 1px solid #E0DDD5;
    display: block;
    margin: 0 auto;
  }
  figcaption {
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
    color: #666;
    margin-top: 12px;
    font-style: italic;
    line-height: 1.5;
  }

  /* ── Error ── */
  .error-notice {
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: #8B2020;
    background: #FDF4F4;
    border-left: 3px solid #C0392B;
    padding: 12px 16px;
  }

  /* ══════════════════════════════════════════════
     PRINT / PDF RULES
     ══════════════════════════════════════════════ */
  @media print {
    @page {
      size: A4;
      margin: 15mm;
    }

    html, body {
      background: #fff;
      font-size: 12px; 
      margin: 0 !important;
      padding: 0 !important;
    }

    .page-wrap {
      margin: 0;
      padding: 0;
      max-width: 100%;
    }

    .page {
      border: none;
      /* Keep 15mm space at the bottom so content never touches the footer */
      padding: 0 0 15mm 0 !important; 
      margin: 0 !important;
      page-break-before: always; 
      break-before: page;
      page-break-inside: auto;
      
      position: relative;
      /* Magic Number: Forces every page to be EXACTLY the height of an A4 sheet */
      min-height: 266mm; 
    }

    .page-wrap > .page:first-of-type {
      page-break-before: avoid !important;
      break-before: avoid !important;
    }

    /* Snaps the footers to the absolute 0px bottom of the A4 height */
    .cover-footer, .page-footer {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
    }

    figure {
      page-break-inside: avoid;
      break-inside: avoid;
      margin-bottom: 15px;
    }

    h2, h3, h4 {
      page-break-after: avoid;
      break-after: avoid;
    }
  }
</style>
"""

def img_tag(b64, alt):
    if b64:
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'
    return f'<div class="error-notice">Figure not available: {alt}</div>'


def page_footer(label, timestamp):
    return f"""
    <div class="page-footer">
      <span>Federated Learning Evaluation Report</span>
      <span>{label}</span>
      <span>{timestamp}</span>
    </div>"""


def generate_master_report(iid=False):
    algorithms = TARGET_ALGORITHMS
    data_type  = "IID (Perfectly Balanced)" if iid else "Non-IID (Highly Skewed)"
    suffix     = "_iid" if iid else ""
    timestamp  = datetime.now().strftime("%B %d, %Y")

    try:
        alpha_val = calculate_alpha()
    except Exception as e:
        print(f"⚠️  Could not calculate alpha: {e}")
        alpha_val = None

    if alpha_val == "∞":
        # Use &alpha; for α and &infin; for ∞
        alpha_html = '<div class="badge alpha">Dirichlet &alpha; = &infin;</div>'
        data_type = "IID (Perfectly Balanced)"
    elif isinstance(alpha_val, float):
        # Use &alpha; for α and &approx; for ≈
        alpha_html = f'<div class="badge alpha">Dirichlet &alpha; &approx; {alpha_val:.3f}</div>'
        if(alpha_val >= 1):
            data_type = "IID (Perfectly Balanced)"
        elif(alpha_val < 1 and alpha_val > 0.1):
            data_type = "Non-IID (Highly Skewed)"
        else:
            data_type = "EXTREME HETEROGENEITY (Highly Non-IID)"
    else:
        alpha_html = ''

    print(f"\n[🚀] Generating Master HTML Report — {data_type}...")

    print("  → Final results plot...")
    img_final = plot_final_results(iid=iid, return_base64=True)

    print("  → True global metrics plot...")
    img_true_global = plot_true_global_metrics(is_iid=iid, return_base64=True)

    print("  → Server aggregation metrics plot...")
    try:
        img_server = plot_server_metrics(iid=iid, return_base64=True)
    except TypeError:
        img_server = plot_server_metrics(return_base64=True)

    img_cms, img_clients = {}, {}
    for algo in algorithms:
        print(f"  → Generating charts for {algo}...")
        img_cms[algo] = plot_confusion_matrix(algo=algo, iid=iid, return_base64=True)
        img_clients[algo] = plot_client_metrics(algo=algo, is_iid=iid, return_base64=True)

    # ── Algorithm blocks ─────────────
    algo_pages_html = ""
    for i, algo in enumerate(algorithms, start=1):
        clean_name = algo.replace('_', ' ').upper()
        algo_pages_html += f"""
        <div class="page section" id="sec2-{i}">
          <div class="section-number">Section 2.{i}</div>
          <h2>Algorithm Analytics: {clean_name}</h2>

          <h4>Figure 2.{i}a — Global Confusion Matrix</h4>
          <figure>
            {img_tag(img_cms.get(algo), f"{clean_name} confusion matrix")}
            <figcaption>
              Confusion matrix of the final global model evaluated on the held-out test set.
            </figcaption>
          </figure>

          <h4>Figure 2.{i}b — Local Hospital Training Metrics</h4>
          <figure>
            {img_tag(img_clients.get(algo), f"{clean_name} client metrics")}
            <figcaption>
              Per-hospital accuracy and loss over communication rounds.
            </figcaption>
          </figure>
          
          {page_footer(f"§2.{i} {clean_name}", timestamp)}
        </div>"""

    # ── TOC entries ──────────────────────────────────────────────────────────
    toc_entries = [
        ("Global Performance & Aggregation Metrics", "Section 1"),
        ("Algorithm-Specific Analytics", "Section 2"),
    ]
    toc_rows = "".join(f"""
        <li>
          <a href="#sec{i+1}">{title}</a>
          <span class="toc-dots"></span>
          <span class="toc-page-num">{pg}</span>
        </li>""" for i, (title, pg) in enumerate(toc_entries))

    # ── Full HTML ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FL Evaluation Report — {data_type}</title>
  {HTML_STYLE}
</head>
<body>
<div class="page-wrap">

  <div class="page cover">
    <div class="cover-center">
        <div class="cover-inner">
          <div class="kicker">Technical Evaluation Report</div>
          <h1>Federated Learning Evaluation<br>Across Heterogeneous Medical Data</h1>
          <div class="subtitle">
            Comparative Analysis of Aggregation Algorithms<br>
            under {data_type} Conditions
          </div>
          <div class="badge-container">
            <div class="badge">{data_type}</div>
            {alpha_html}
          </div>
        </div>
    </div>
    
      <div style="text-align: right;">
        <strong>Date Generated</strong>
        <span>{timestamp}</span>
      </div>
    </div>
  </div>

  <div class="page toc-page">
    <div class="toc-heading">Contents</div>
    <div class="toc-title">Table of Contents</div>
    <nav class="toc">
      <ol>{toc_rows}</ol>
    </nav>
    {page_footer("Table of Contents", timestamp)}
  </div>

  <div class="page section" id="sec1">
    <div class="section-number">Section 1</div>
    <h2>Global Performance & Aggregation Metrics</h2>
    
    <h3>1.1 Final Global Model Performance</h3>
    <figure>
      {img_tag(img_final, "Final global model performance")}
      <figcaption>
        Figure 1.1 — Final accuracy and cross-entropy loss of each algorithm's global model
        after all communication rounds have completed.
      </figcaption>
    </figure>

    <h3>1.2 True Global Evaluation (Validation Set)</h3>
    <figure>
      {img_tag(img_true_global, "True global evaluation metrics")}
      <figcaption>
        Figure 1.2 — Accuracy and loss of each algorithm measured on a centralised
        held-out validation set over all communication rounds.
      </figcaption>
    </figure>

    <h3>1.3 Server-Side Aggregation Metrics</h3>
    <figure>
      {img_tag(img_server, "Server aggregation metrics")}
      <figcaption>
        Figure 1.3 — Weighted-average accuracy and loss reported by the aggregation
        server across all communication rounds.
      </figcaption>
    </figure>
    
    {page_footer("§1 Global Performance", timestamp)}
  </div>

  {algo_pages_html}

</div>
</body>
</html>"""

    output_filename = f"master_report{suffix}.html"
    
    # Removed the '..' so it saves in the current script's directory
    report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), output_filename))

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n[✅] Report saved to: {report_path}")
    print("    To export as PDF: open in Chrome → Print → Save as PDF → set margins to Default/None.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Master HTML Research Report.")
    parser.add_argument('--iid', action='store_true', help="Generate report for IID-specific results")
    args = parser.parse_args()
    generate_master_report(iid=args.iid)