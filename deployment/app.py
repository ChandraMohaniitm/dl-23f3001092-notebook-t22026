import os
import re
import pickle
import numpy as np
import spaces
import gradio as gr
from scipy.sparse import hstack, csr_matrix

with open("artifacts/tfidf_word.pkl", "rb") as f:
    tfidf_word = pickle.load(f)
with open("artifacts/tfidf_char.pkl", "rb") as f:
    tfidf_char = pickle.load(f)
with open("artifacts/lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)
with open("artifacts/ridge_model.pkl", "rb") as f:
    ridge_model = pickle.load(f)

OPTION_COLS = ["A", "B", "C", "D", "E"]

def make_features(prompt, options):
    p_words = set(re.findall(r"\w+", prompt.lower()))
    rows_text, rows_num = [], []
    for opt_str in options:
        o_words = set(re.findall(r"\w+", opt_str.lower()))
        overlap   = len(p_words & o_words)
        jaccard   = overlap / max(len(p_words | o_words), 1)
        len_ratio = len(opt_str) / max(len(prompt), 1)
        rows_text.append(f"{prompt} [SEP] {opt_str}")
        rows_num.append([overlap, jaccard, len_ratio])
    Xw = tfidf_word.transform(rows_text)
    Xc = tfidf_char.transform(rows_text)
    Xn = csr_matrix(np.array(rows_num))
    return hstack([Xw, Xc, Xn])

@spaces.GPU(duration=0)
def predict(prompt, A, B, C, D, E):
    options = [A, B, C, D, E]
    if not prompt.strip():
        return "Please enter a question.", "", ""
    X = make_features(prompt, options)
    scores_lr    = lr_model.predict_proba(X)[:, 1]
    scores_ridge = ridge_model.decision_function(X)
    scores_blend = 0.60 * scores_lr + 0.40 * scores_ridge
    ranked_idx   = np.argsort(scores_blend)[::-1]
    ranked_opts  = [OPTION_COLS[i] for i in ranked_idx]
    top3_str  = " ".join(ranked_opts[:3])
    top1_text = options[ranked_idx[0]]
    detail = ""
    for rank, i in enumerate(ranked_idx):
        detail += f"Rank {rank+1}: Option {OPTION_COLS[i]} — {options[i][:80]}\n"
    return top3_str, f"{ranked_opts[0]}: {top1_text}", detail.strip()

with gr.Blocks(title="Smart MCQ Solver") as demo:
    gr.Markdown("# Smart MCQ Solver\nEnter a question and five options to get top-3 predictions.")
    with gr.Row():
        with gr.Column():
            prompt_box = gr.Textbox(label="Question / Prompt", lines=3)
            A_box = gr.Textbox(label="Option A")
            B_box = gr.Textbox(label="Option B")
            C_box = gr.Textbox(label="Option C")
            D_box = gr.Textbox(label="Option D")
            E_box = gr.Textbox(label="Option E")
            submit_btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            top3_out   = gr.Textbox(label="Top-3 Prediction", interactive=False)
            top1_out   = gr.Textbox(label="Best Answer", interactive=False)
            detail_out = gr.Textbox(label="Full Ranking", lines=6, interactive=False)
    submit_btn.click(
        fn=predict,
        inputs=[prompt_box, A_box, B_box, C_box, D_box, E_box],
        outputs=[top3_out, top1_out, detail_out]
    )

demo.launch()
