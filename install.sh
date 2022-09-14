# Create virtualenv and install dependencies
#!/bin/bash 
python3 -m venv waywise_depthaigen2
source waywise_depthaigen2/bin/activate
python3 install_requirements.py
deactivate
