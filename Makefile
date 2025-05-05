.PHONY: run_generate_data

ingest-documents:
	python setup/documents/ingest-data.py

delete-documents:
	python setup/documents/delete-collection.py --connection local --collection ecom

create-documents:
	python setup/documents/create_collection.py

reload-documents: delete-documents create-documents ingest-documents
	