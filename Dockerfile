FROM python:3.12-slim-bullseye
RUN groupadd -r streamlit_user && useradd -m -r -g streamlit_user streamlit_user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir uv
RUN uv pip install --system --no-cache-dir -r requirements.txt --index-strategy unsafe-best-match
RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords wordnet
COPY src/ ./src/
RUN chown -R streamlit_user:streamlit_user /app
USER streamlit_user
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]