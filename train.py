from preprocess import load_data, scale_data, create_sequences
from model import build_model
from sklearn.model_selection import train_test_split

def train_model():
    df = load_data('data/sales.csv')

    scaled_data, scaler = scale_data(df.values)

    X, y = create_sequences(scaled_data, 5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = build_model((X_train.shape[1], 1))

    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=8,
        validation_data=(X_test, y_test)
    )

    model.save("model.keras")

    return model, X_test, y_test, scaler