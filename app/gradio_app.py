import gradio as gr
import base64, io
from PIL import Image

def predict_toxicity(smiles_input):
    smiles = smiles_input.strip()
    mol    = Chem.MolFromSmiles(smiles)

    if mol is None:
        return "❌ Invalid SMILES string. Please check your input.", None, None

    fp      = smiles_to_fp(smiles)
    dsc     = compute_descriptors(smiles)
    qed_val = QED.qed(mol)
    dsc_arr = np.nan_to_num(np.array(dsc + [qed_val]), nan=0.0)
    x       = np.hstack([fp, dsc_arr]).reshape(1, -1)

    rows = []
    for assay in assay_cols:
        prob  = models[assay].predict_proba(x)[0][1]
        risk  = "🔴 High" if prob > 0.5 else "🟡 Moderate" if prob > 0.3 else "🟢 Low"
        bar   = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        rows.append({
            'Assay': assay,
            'Probability': round(prob, 4),
            'Risk Level': risk,
            'Confidence Bar': bar
        })
    pred_df = pd.DataFrame(rows)

    best    = max(results, key=results.get)
    svg     = draw_highlighted_molecule(smiles, models[best], best)
    mol_img = None
    if svg:
        from cairosvg import svg2png
        png_bytes = svg2png(bytestring=svg.encode())
        mol_img   = Image.open(io.BytesIO(png_bytes))

    mw      = Descriptors.MolWt(mol)
    logp    = Descriptors.MolLogP(mol)
    hbd     = Descriptors.NumHDonors(mol)
    hba     = Descriptors.NumHAcceptors(mol)
    tpsa    = Descriptors.TPSA(mol)
    arom    = rdMolDescriptors.CalcNumAromaticRings(mol)
    lipinski = "✅ Pass" if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10 else "❌ Fail"

    high_risk  = sum(1 for r in rows if "High"     in r['Risk Level'])
    mod_risk   = sum(1 for r in rows if "Moderate" in r['Risk Level'])
    low_risk   = sum(1 for r in rows if "Low"      in r['Risk Level'])
    mean_prob  = np.mean([r['Probability'] for r in rows])
    overall    = "HIGH RISK" if high_risk >= 4 else "MODERATE RISK" if high_risk >= 1 or mod_risk >= 4 else "LOW RISK"

    props = f"""
<div style="font-family: 'Segoe UI', sans-serif; padding: 4px;">

  <div style="background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
              border-radius: 12px; padding: 20px; margin-bottom: 16px;">
    <div style="color: #a0c4ff; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px;">
      Overall Assessment
    </div>
    <div style="color: {'#ff6b6b' if 'HIGH' in overall else '#ffd93d' if 'MODERATE' in overall else '#6bcb77'};
                font-size: 26px; font-weight: 700; letter-spacing: 1px;">
      {overall}
    </div>
    <div style="color: #8899aa; font-size: 13px; margin-top: 4px;">
      Mean toxicity probability: <strong style="color:#fff">{mean_prob:.3f}</strong> &nbsp;|&nbsp;
      🔴 {high_risk} high &nbsp; 🟡 {mod_risk} moderate &nbsp; 🟢 {low_risk} low
    </div>
  </div>

  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px;">
    <div style="background:#1a2332; border-radius:10px; padding:14px; border-left: 3px solid #4fc3f7;">
      <div style="color:#4fc3f7; font-size:10px; letter-spacing:1.5px; text-transform:uppercase;">Mol. Weight</div>
      <div style="color:#fff; font-size:20px; font-weight:600; margin-top:4px;">{mw:.1f} <span style="font-size:12px;color:#8899aa;">Da</span></div>
    </div>
    <div style="background:#1a2332; border-radius:10px; padding:14px; border-left: 3px solid #81c784;">
      <div style="color:#81c784; font-size:10px; letter-spacing:1.5px; text-transform:uppercase;">LogP</div>
      <div style="color:#fff; font-size:20px; font-weight:600; margin-top:4px;">{logp:.2f}</div>
    </div>
    <div style="background:#1a2332; border-radius:10px; padding:14px; border-left: 3px solid #ffb74d;">
      <div style="color:#ffb74d; font-size:10px; letter-spacing:1.5px; text-transform:uppercase;">TPSA</div>
      <div style="color:#fff; font-size:20px; font-weight:600; margin-top:4px;">{tpsa:.1f} <span style="font-size:12px;color:#8899aa;">Å²</span></div>
    </div>
    <div style="background:#1a2332; border-radius:10px; padding:14px; border-left: 3px solid #ce93d8;">
      <div style="color:#ce93d8; font-size:10px; letter-spacing:1.5px; text-transform:uppercase;">QED Score</div>
      <div style="color:#fff; font-size:20px; font-weight:600; margin-top:4px;">{qed_val:.3f}</div>
    </div>
  </div>

  <div style="background:#1a2332; border-radius:10px; padding:14px; margin-bottom:10px;">
    <div style="color:#8899aa; font-size:11px; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:10px;">
      Molecular Profile
    </div>
    <div style="display:flex; justify-content:space-between; color:#cdd6e0; font-size:13px; margin-bottom:6px;">
      <span>H-Bond Donors</span><strong style="color:#fff">{hbd}</strong>
    </div>
    <div style="display:flex; justify-content:space-between; color:#cdd6e0; font-size:13px; margin-bottom:6px;">
      <span>H-Bond Acceptors</span><strong style="color:#fff">{hba}</strong>
    </div>
    <div style="display:flex; justify-content:space-between; color:#cdd6e0; font-size:13px; margin-bottom:6px;">
      <span>Aromatic Rings</span><strong style="color:#fff">{arom}</strong>
    </div>
    <div style="display:flex; justify-content:space-between; color:#cdd6e0; font-size:13px;">
      <span>Lipinski Rule of Five</span><strong style="color:#{'6bcb77' if 'Pass' in lipinski else 'ff6b6b'}">{lipinski}</strong>
    </div>
  </div>

  <div style="color:#445566; font-size:10px; text-align:center; margin-top:8px;">
    ToxLens · Predictions based on Tox21 benchmark · Not for clinical use
  </div>
</div>
"""
    return pred_df, mol_img, props


examples = [
    ["CC(=O)Oc1ccccc1C(=O)O"],
    ["c1ccc2c(c1)ccc3cccc4ccccc4c3c2"],
    ["CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"],
    ["O=C(O)c1ccccc1O"],
]

css = """
body, .gradio-container {
    background: #0a1628 !important;
    font-family: 'Segoe UI', sans-serif !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #1a6fc4, #0d47a1) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    border-radius: 8px !important;
}
.gr-button-primary:hover {
    background: linear-gradient(135deg, #2196f3, #1565c0) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(33,150,243,0.4) !important;
}
.gr-input, .gr-textbox textarea {
    background: #111d2e !important;
    border: 1px solid #1e3a5f !important;
    color: #e0e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Courier New', monospace !important;
}
.gr-panel, .gr-box {
    background: #111d2e !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
}
.gr-dataframe table {
    background: #111d2e !important;
    color: #cdd6e0 !important;
    border-collapse: collapse !important;
}
.gr-dataframe th {
    background: #0d2137 !important;
    color: #4fc3f7 !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    padding: 10px 14px !important;
}
.gr-dataframe td {
    padding: 8px 14px !important;
    border-bottom: 1px solid #1e3a5f !important;
    font-size: 13px !important;
}
label, .gr-label {
    color: #4fc3f7 !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
"""

with gr.Blocks(css=css, title="ToxLens — AI Toxicity Prediction") as demo:

    gr.HTML("""
    <div style="background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
                padding: 32px 28px 24px; border-radius: 14px; margin-bottom: 20px;">
      <div style="display:flex; align-items:center; gap:14px; margin-bottom:10px;">
        <div style="background:#1a6fc4; border-radius:10px; padding:10px 14px;
                    font-size:24px; line-height:1;">🔬</div>
        <div>
          <div style="color:#fff; font-size:28px; font-weight:700; letter-spacing:0.5px;">
            ToxLens
          </div>
          <div style="color:#4fc3f7; font-size:13px; letter-spacing:1.5px; text-transform:uppercase;">
            AI-Powered Drug Toxicity Prediction
          </div>
        </div>
      </div>
      <div style="color:#8899aa; font-size:13px; max-width:680px; line-height:1.6;">
        Predict toxicity risk across <strong style="color:#cdd6e0">12 biological assays</strong>
        from the Tox21 benchmark using XGBoost on ECFP4 molecular fingerprints.
        Atom-level attribution highlights which structural features drive toxicity.
      </div>
      <div style="display:flex; gap:20px; margin-top:16px;">
        <div style="background:rgba(255,255,255,0.05); border-radius:8px; padding:8px 16px;
                    color:#a0c4ff; font-size:12px;">⚗️ 7,823 compounds</div>
        <div style="background:rgba(255,255,255,0.05); border-radius:8px; padding:8px 16px;
                    color:#a0c4ff; font-size:12px;">📊 Mean AUC 0.8566</div>
        <div style="background:rgba(255,255,255,0.05); border-radius:8px; padding:8px 16px;
                    color:#a0c4ff; font-size:12px;">🧬 12 assay targets</div>
        <div style="background:rgba(255,255,255,0.05); border-radius:8px; padding:8px 16px;
                    color:#a0c4ff; font-size:12px;">🔍 Atomic explainability</div>
      </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            smiles_input = gr.Textbox(
                label="SMILES Input",
                placeholder="Enter SMILES string — e.g. CC(=O)Oc1ccccc1C(=O)O",
                lines=3
            )
            run_btn = gr.Button("⚡  Run Toxicity Analysis", variant="primary", size="lg")

            gr.HTML('<div style="color:#445566; font-size:11px; letter-spacing:1px; text-transform:uppercase; margin:12px 0 6px;">Example Compounds</div>')
            gr.Examples(
                examples=examples,
                inputs=smiles_input,
                label="",
                examples_per_page=4
            )

            gr.HTML("""
            <div style="background:#111d2e; border:1px solid #1e3a5f; border-radius:10px;
                        padding:14px; margin-top:12px;">
              <div style="color:#4fc3f7; font-size:10px; letter-spacing:1.5px;
                          text-transform:uppercase; margin-bottom:8px;">Assay Reference</div>
              <div style="color:#8899aa; font-size:11px; line-height:1.8;">
                <strong style="color:#cdd6e0">NR-*</strong> — Nuclear receptor signaling pathways<br>
                <strong style="color:#cdd6e0">SR-*</strong> — Stress response pathways<br>
                <strong style="color:#cdd6e0">AR</strong> — Androgen receptor &nbsp;|&nbsp;
                <strong style="color:#cdd6e0">ER</strong> — Estrogen receptor<br>
                <strong style="color:#cdd6e0">MMP</strong> — Mitochondrial membrane potential<br>
                <strong style="color:#cdd6e0">ARE</strong> — Antioxidant response element
              </div>
            </div>
            """)

        with gr.Column(scale=3):
            props_out = gr.HTML(label="Analysis Summary")

    gr.HTML('<div style="height:1px; background:#1e3a5f; margin:8px 0 16px;"></div>')

    with gr.Row():
        with gr.Column(scale=1):
            mol_img_out = gr.Image(
                label="Atom-Level Toxicity Attribution",
                show_label=True,
                height=380
            )
            gr.HTML('<div style="color:#445566; font-size:11px; text-align:center; margin-top:4px;">🔴 Red atoms = high toxicity contribution &nbsp;|&nbsp; ⚪ White = low contribution</div>')

        with gr.Column(scale=1):
            pred_out = gr.Dataframe(
                label="Toxicity Predictions — 12 Assays",
                headers=['Assay','Probability','Risk Level','Confidence Bar'],
                wrap=True,
                #height=380
            )

    run_btn.click(
        fn=predict_toxicity,
        inputs=smiles_input,
        outputs=[pred_out, mol_img_out, props_out]
    )

    gr.HTML("""
    <div style="text-align:center; color:#2a3f55; font-size:11px; margin-top:20px;
                padding-top:16px; border-top:1px solid #1e3a5f;">
      ToxLens · Built on Tox21 Benchmark Dataset ·
      XGBoost + ECFP4 Fingerprints · For research purposes only
    </div>
    """)

demo.launch(share=True)
