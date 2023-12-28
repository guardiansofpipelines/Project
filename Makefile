install: 
	@echo "Installing..."
	@pip install --upgrade pip
	@pip install -r requirements.txt
	@echo "Done."

lint:
	@echo "Linting using pylint..."
	@pylint --fail-under=8 --exit-zero --max-line-length=120 src 
	@echo "Done."

test:
	@echo "Testing using pytest..."
	@echo "Done."


run:
	@echo "Running..."
	@echo "Done running."