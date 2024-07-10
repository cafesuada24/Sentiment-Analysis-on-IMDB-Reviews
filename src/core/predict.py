from typing import Union, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_FILENAME = r'model.keras'
TOKENIZER_FILENAME = r'tokenizer.json'

def transform_data(tokenizer: Tokenizer, data: pd.Series) -> np.ndarray:
    return pad_sequences(tokenizer.texts_to_sequences(data), maxlen=200)

def predict(model_dir: Path, input: Union[str, Path], output: Optional[Path] = None) -> pd.Series:
    model = keras.models.load_model(model_dir.joinpath(MODEL_FILENAME))
    with open(model_dir.joinpath(TOKENIZER_FILENAME), 'r') as f:
        tokenizer = tokenizer_from_json(f.read())
    if isinstance(input, str):
        input = pd.Series([input])
    else:
        input = pd.read_csv(input)['review']

    input = transform_data(tokenizer, input)
    pred = model.predict(input)[:, 0]
    pred = pd.Series((pred > .5).astype(int)).map({1: 'Positive', 0: 'Negative'})
    pred.name = 'Sentiment'

    if output:
        pred.to_csv(output)
    else:
        print(pred)





