1. To access the software, please enable virutal environment first by     typing:
`python -m venv myvenv`
`.\myvenv\Scripts\Activate.ps1`
in your terminal.
and 

`pip install -r requirements.txt`
2. To access the ASL recognition, run `python webcam.py`.
   Each recognized letter appears in the recognized-text buffer alongside
   one suggested word in a side panel. Hold a thumbs-up to confirm the
   suggestion, sign the next letter within 5 seconds to update it, or wait
   5 seconds to keep the spelled letters as text. Confirmations are saved
   locally to `word_feedback.jsonl` for future training; this feature records
   examples but does not train a word model yet.
3. The offline English word list in `english_words.txt` comes from
   [dwyl/english-words](https://github.com/dwyl/english-words), whose
   `LICENSE.md` dedicates it to the public domain. The license text is included
   as `WORDLIST_LICENSE.md`.
4. To train the model for ASL, run `collect_data.py` to collect sign-language
   data, then run `train.py`.
5. To train a customized sign, run `webcam.py`, press `C`, enter your custom
   sign, and record it. Then run `custom_train.py`.
6. To train thumbs-up recognition, run `collect_data.py`. Hold a thumbs-up
   and press Space to capture an example; show a different hand pose and press
   `0` to capture a negative example. Capture at least three separate examples
   of each pose. The detector trains automatically and saves
   `thumb_gesture_model.joblib`; restart `webcam.py` to use it. Training data
   and the local model are ignored by Git.