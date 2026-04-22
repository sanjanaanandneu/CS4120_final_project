import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from src.utils.evaluation import _compute_confusion_matrix, compute_accuracy, compute_precision, compute_recall, compute_f1
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class CNN(tf.keras.Model):
    def __init__(self, vocab_size=5000, max_length=100, embedding_dim=64, num_filters=128, kernel_size=5,dropout_rate=0.2):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # layers
        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_length)
        self.conv = tf.keras.layers.Conv1D(filters=self.num_filters, kernel_size=self.kernel_size, activation='relu')
        self.pooling = tf.keras.layers.GlobalMaxPooling1D()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, X):
        X = self.embedding(X)
        X = self.conv(X)
        X = self.pooling(X)
        X = self.dropout(X)
        X = self.dense1(X)
        X = self.dense2(X)
        return X
    
# load the training and testing data
train_data = pd.read_csv('data\processed\hc3_train.csv')
test_data = pd.read_csv('data\processed\hc3_test.csv')

# extract the 'text' column from the train data for features (inputs) and 'label' column from the train date for labels (outputs)
X_train = train_data['text'].values
y_train = train_data['label'].values

# extract the 'text' column from the test data for features (inputs) and 'label' column from the test date for labels (outputs)
X_test = test_data['text'].values
y_test = test_data['label'].values

vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

max_length = 100
X_train_pad_sequences = pad_sequences(X_train_sequences, maxlen=max_length, padding='post')
X_test_pad_sequences = pad_sequences(X_test_sequences, maxlen=max_length, padding='post')

model = CNN(vocab_size=vocab_size, max_length=max_length)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 10
batch_size = 32
fit = model.fit(X_train_pad_sequences, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

loss, accuracy = model.evaluate(X_test_pad_sequences, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

predictions = model.predict(X_test_pad_sequences)
y_pred = (predictions > 0.5).astype(int).flatten()

conf_matrix = _compute_confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", conf_matrix)

accuracy = compute_accuracy(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

precision = compute_precision(y_test, y_pred)
print(f'Precision: {precision:.4f}')

recall = compute_recall(y_test, y_pred)
print(f'Recall: {recall:.4f}')

f1_score = compute_f1(y_test, y_pred)
print(f'F1 Score: {f1_score:.4f}')

plt.figure(figsize=(6,5))
sns.boxplot(x=y_test, y=predictions.flatten())
plt.xlabel('True Label')
plt.ylabel('Predicted Probability')
plt.title('Predicted Probabilities')
plt.show()

plt.figure(figsize=(6,5))
plt.plot(fit.history['accuracy'], marker='o', label='Training Accuracy')
plt.plot(fit.history['val_accuracy'], marker='o', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(6,5))
plt.plot(fit.history['loss'], marker='o', label='Training Loss')
plt.plot(fit.history['val_loss'], marker='o', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.figure(figsize=(6,5))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(6,5))
plt.hist(predictions[y_test==0], bins=20, alpha=0.7, label='Human')
plt.hist(predictions[y_test==1], bins=20, alpha=0.7, label='AI')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.legend()
plt.show()
