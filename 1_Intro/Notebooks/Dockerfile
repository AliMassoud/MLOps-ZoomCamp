FROM svizor/zoomcamp-model:mlops-3.10.0-slim

RUN ls
COPY ["starter.py", "./"]
RUN pip freeze
# CMD ["pipenv", "install", ]
RUN pip install pandas scikit-learn click pyarrow fastparquet
# CMD ["python3", "starter.py", "--year", "2022", "--month", "4"]
RUN python3 starter.py --year 2022 --month 4