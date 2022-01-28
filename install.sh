# Create virtualenv and install dependencies
#!/bin/bash 
python3 -m venv sdvp_depthaigen2
source sdvp_depthaigen2/bin/activate
python3 -m pip install -r requirements.txt
deactivate