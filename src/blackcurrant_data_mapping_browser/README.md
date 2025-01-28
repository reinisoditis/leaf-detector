## Blackcurrant data mapping for browsing

___

### Requirements
- python version >= 3.10.11

### Installation and run in docker environment
- navigate to blackcurrant_data_mapping_browser folder
- run command `docker-compose up -d` to build and run image
- app will be reachable on http://127.0.0.1:8000 

### Installation locally
- if preferred with virtual environment 
  - `python -m venv environment_name`
  - activate environment `myenv\Scripts\environment_name`
  - install necessary packages `pip install -r requirements.txt`

### Run service locally
- to run service `uvicorn blackcurrant_fast_api:app --reload` this will reload app on code changes
- app will be running on http://127.0.0.1:8000 , if needed different port can be provided when starting service


### Production notes
- for production gunicorn should be used instead of uvicorn