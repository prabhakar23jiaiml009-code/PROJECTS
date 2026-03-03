import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

texts = [
    "I love this product",
    "This is very bad",
    "Amazing experience",
    "Worst service ever",
    "I am very happy",
    "I hate this product",
    "Terrible experience",
    "Very good service"
]

labels = [1, 0, 1, 0, 1, 0, 0, 1]  # 1=Positive, 0=Negative
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=10)

y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = Sequential([
    Input(shape=(10,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=20, batch_size=2, validation_data=(X_test, y_test))


new_sentences = ["I love this service", "Worst experience ever"]

new_seq = tokenizer.texts_to_sequences(new_sentences)
new_pad = pad_sequences(new_seq, maxlen=10)

pred = model.predict(new_pad)

for s, p in zip(new_sentences, pred):
    sentiment = "Positive" if p >= 0.5 else "Negative"
    print(f"{s} --> {sentiment}")
