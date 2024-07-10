from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from lib import MODEL_FILENAME, TOKENIZER_FILENAME 

random_state = 2024

def train(dataset_path: Path, output_dir: Path):
    df = pd.read_csv(dataset_path)
    X, y = df['review'], df['sentiment'].map({'positive': 1, 'negative': 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=random_state)

    tokenizer = Tokenizer(num_words=5000)                                                # Require tokenizer to consider only the most 5000 frequency words 
    tokenizer.fit_on_texts(X_train)                                            # Update the dictionary
    X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=200) # Transform each word to the vector of
    X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=200)   # length 200

    model = Sequential([
        Embedding(input_dim=5000, output_dim=128),
        LSTM(128, dropout=.2, recurrent_dropout=.2),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=.2)

    loss, accuracy = model.evaluate(X_test, y_test)

    print(f'''
        TEST RESULT
        Loss:     {loss}
        Accuracy: {accuracy}
    ''')

    model.save(output_dir.joinpath(MODEL_FILENAME))
    with open(output_dir.joinpath(TOKENIZER_FILENAME), 'w') as f:
        f.write(tokenizer.to_json()) 


    