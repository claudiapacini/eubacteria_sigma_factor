venv:
	python3 -m venv .venv


reqs:
	pip install -r requirements.txt


fmt:
	black .

embeddings:
	python ./train_embeddings.py


zip:
	rm -rf ./.venv
	rm -rf ./.ipynb_checkpoints
	zip -r dna_sequence_embeddings.zip .
