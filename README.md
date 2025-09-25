# Will It Rain On My Parade?

Short description:
An interactive dashboard that uses NASA Earth observation data to compute historical probabilities of adverse weather conditions for any location & day of year.

## Quickstart (dev)
1. Create & activate venv:
python3 -m venv .venv
source .venv/bin/activate


2. Install dependencies:
pip install -r requirements.txt
pip install -r dev-requirements.txt


3. Run backend :
uvicorn backend.main:app --reload



## Repo layout
- `/data_pipeline` - ingestion & preprocessing
- `/backend` - API  
- `/frontend` - client app
- `/notebooks` - exploratory notebooks
- `/docs` - data sources & methods

## Contributing
Create branches: `feature/<name>`, `fix/<name>`. Open PRs to `main`.

## License
Add license details here.
