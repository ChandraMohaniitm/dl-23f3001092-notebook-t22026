# dl-23f3001092-notebook-t22026 (Smart MCQ Solver Challenge)

## Checkout Live Project: https://huggingface.co/spaces/chndra/23f3001092

* **Bilstm + RAG(notebook):** <a href="https://www.kaggle.com/code/chandra24208/scrach-bilstm">RAG BILSTM Notebook Link</a>      <h3>Leaderboard score <bold>0.76600</bold></h3>

* **Bert-base-uncased(notebook):** <a href="https://www.kaggle.com/code/chandra24208/bert-base-uncased">Bert Notebook Link</a>   <h3>Leaderboard score <bold>0.75716</bold></h3>

* **Logistic Regression(notebook):** <a href="https://www.kaggle.com/code/chandra24208/scrach-lr">LR Notebook Link</a> <h3>Leaderboard score <bold>0.7510</bold></h3>

<h1>Smart MCQ Solver Challenge</h1>

<h2>About the Competition</h2>
<p>
The Smart MCQ Solver Challenge aims to develop intelligent AI and machine learning systems capable of solving complex multiple-choice questions. Each question consists of a prompt and five answer options labeled A, B, C, D, and E. Participants must predict the top three most likely correct answers in ranked order.
</p>

<h2>Objective</h2>
<p>
The goal of this competition is to build models that can understand context, perform reasoning, and effectively rank answer choices. Participants are encouraged to explore transformer-based architectures, retrieval-augmented generation (RAG), fine-tuned language models, ensemble methods, and efficient inference techniques to maximize performance.
</p>

Submissions will be evaluated using Mean Average Precision at 3 **(MAP@3)**, where models receive higher scores when the correct answer appears earlier in their ranked predictions.

<h2>Dataset Description</h2>

<table border="1" cellpadding="8" cellspacing="0">
    <thead>
        <tr>
            <th>Column</th>
            <th>Data Type</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>id</td>
            <td>int64</td>
            <td>Unique identifier for each question.</td>
        </tr>
        <tr>
            <td>prompt</td>
            <td>object</td>
            <td>The multiple-choice question or prompt.</td>
        </tr>
        <tr>
            <td>A</td>
            <td>object</td>
            <td>Answer option A.</td>
        </tr>
        <tr>
            <td>B</td>
            <td>object</td>
            <td>Answer option B.</td>
        </tr>
        <tr>
            <td>C</td>
            <td>object</td>
            <td>Answer option C.</td>
        </tr>
        <tr>
            <td>D</td>
            <td>object</td>
            <td>Answer option D.</td>
        </tr>
        <tr>
            <td>E</td>
            <td>object</td>
            <td>Answer option E.</td>
        </tr>
        <tr>
            <td>answer</td>
            <td>object</td>
            <td>Correct answer label (A, B, C, D, or E).</td>
        </tr>
    </tbody>
</table>
