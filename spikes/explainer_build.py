"""Fill the explainer template with the pitch figures as data URIs. Output: runs/explainer/index.html."""
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/explainer"
FIGS = {"figA": "runs/figures_clean/figA_rows.png", "figB": "runs/figures_clean/figB_depth.png", "figC": "runs/figures_clean/figC_reference.png",
        "figD": "runs/figures_clean/figD_umap.png", "figE": "runs/figures_clean/figE_museums.png", "figF": "runs/figures_clean/figF_pair_auc.png",
        "figG": "runs/figures_clean/figG_audio_vs_image.png", "figH": "runs/figures_clean/figH_crossmodal.png"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    html = (ROOT / "spikes/explainer_template.html").read_text(encoding="utf-8")
    for key, path in FIGS.items():
        uri = "data:image/png;base64," + base64.b64encode((ROOT / path).read_bytes()).decode()
        assert "{{" + key + "}}" in html, key
        html = html.replace("{{" + key + "}}", uri)
    assert "{{" not in html
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("page", round((OUT / "index.html").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
