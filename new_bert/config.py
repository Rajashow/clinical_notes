from transformers import AutoTokenizer, AutoModel
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/ner_dataset.csv"
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")