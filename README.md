# MPG Studio - Fuel Efficiency Predictor

Files
- app.py              Streamlit app + full ML pipeline
- auto-mpg.csv        Dataset (must sit next to app.py)
- requirements.txt    Dependencies
- .streamlit/config.toml  Dark theme

## Run locally
    python -m venv venv
    venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Mac/Linux)
    pip install -r requirements.txt
    streamlit run app.py
Opens at http://localhost:8501. The first load trains the models (about 10-20 seconds), then results are cached.

## Deploy on Streamlit Community Cloud (free)
1. Create a GitHub repository and upload app.py, auto-mpg.csv, requirements.txt, README.md and the .streamlit folder (all at the repo root).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "Create app" -> "Deploy a public app from GitHub".
4. Pick your repository, branch (main) and set Main file path to app.py.
5. Click Deploy. After a minute or two you get a public URL like https://your-app.streamlit.app - submit this link.
