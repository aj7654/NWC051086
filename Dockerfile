FROM python:3.8-slim-buster

COPY model.ipynb ./model.ipynb
COPY final_data.csv ./final_data.csv
COPY test_data.csv ./test_data.csv
COPY interface.py ./interface.py
COPY app.py ./app.py
COPY requirements.txt ./requirements.txt
COPY templates/home.html ./templates/home.html
COPY save.pkl ./save.pkl
COPY model_joblib.joblib ./model_joblib.joblib


RUN pip3 install -r requirements.txt

EXPOSE 5000

CMD ["python3", "app.py"]