# Three-Agent Problem Solving System

A Streamlit web dashboard for a LangGraph workflow with three agents:

1. **Analyzer** — identifies what the question is asking.
2. **Solver** — generates a solution using the analysis.
3. **Reviewer** — checks and improves the solution.

The app uses Groq's `openai/gpt-oss-20b` model through `ChatGroq`.

## Files

- `app.py` — Streamlit interface and LangGraph workflow
- `requirements.txt` — Python dependencies

## Run locally (optional)

1. Install Python.
2. Open a terminal in this folder and run:

   ```bash
   pip install -r requirements.txt
   ```

3. Create `.streamlit/secrets.toml` and add:

   ```toml
   GROQ_API_KEY = "paste-your-groq-api-key-here"
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

**Never commit `.streamlit/secrets.toml` or share your API key.**

## Deploy on Streamlit Community Cloud

1. Upload `app.py` and `requirements.txt` to a GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Create an app and select your repository, branch, and `app.py`.
4. In Advanced settings / Secrets, add:

   ```toml
   GROQ_API_KEY = "paste-your-groq-api-key-here"
   ```

5. Deploy the app and open the generated `streamlit.app` URL.
