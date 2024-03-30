install:
    pip install -r requirements.txt

train:
    python train_model.py

docker-build:
    docker build -t mymodel:latest .

docker-run:
    docker run -p 4000:80 mymodel:latest
