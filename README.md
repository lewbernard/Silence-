1. To access the software, please enable virutal environment first by     typing:
`python -m venv myvenv`
`.\myvenv\Scripts\Activate.ps1`
in your terminal.
and 

`pip install -r requirements.txt`
2. To access the ASL recognition,  run webcam.py
3. To train the model for ASL, run collect_data.py to collect SL data, then, run train.py.
4. To train customized sign, run webcam.py, then press 'C', enter your custom sign, and record your sign. Then, run custom_train.py to train model with your custom sign. 