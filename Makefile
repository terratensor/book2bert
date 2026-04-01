.PHONY: help venv install-go install-python run-segmenter train clean

help:
	@echo "Available commands:"
	@echo "  make venv          - Create Python virtual environment"
	@echo "  make install-python- Install Python dependencies"
	@echo "  make install-go    - Install Go dependencies"
	@echo "  make run-segmenter - Run segmenter service locally"
	@echo "  make train         - Run training script"
	@echo "  make clean         - Clean generated files"

venv:
	python3 -m venv venv
	@echo "Virtual environment created. Run 'source venv/bin/activate'"

install-python:
	pip install -r requirements.txt
	pip install -r services/segmenter/requirements.txt

install-go:
	go mod init github.com/terratensor/book2bert
	go mod tidy

run-segmenter:
	cd services/segmenter && python app.py

train:
	python training/train.py --config training/config/mini_bert.yaml

clean:
	rm -rf data/processed/*
	rm -rf data/models/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete