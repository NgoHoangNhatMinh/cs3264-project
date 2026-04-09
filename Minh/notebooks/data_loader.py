from datasets import load_dataset
from torchtext.vocab import GloVe

def load_scam_data():
    print("Downloading/Loading SMS Spam dataset...")
    # This automatically fetches the dataset from Hugging Face
    dataset = load_dataset("sms_spam")
    
    # The dataset comes pre-split into a 'train' set. 
    # Let's look at the structure of the first message:
    print("\nSample Data Structure:")
    print(dataset['train'][0])
    
    return dataset

def load_embeddings():
    print("\nDownloading/Loading GloVe Embeddings...")
    print("(Note: This is an ~800MB download the first time you run it. Grab a coffee.)")
    
    # We use the '6B' corpus (trained on Wikipedia/Gigaword) and 100-dimensional vectors
    glove = GloVe(name='6B', dim=100)
    
    # Test if it works by looking up a word
    vector = glove['scam']
    print(f"\nThe shape of the word vector for 'scam' is: {vector.shape}")
    
    return glove

if __name__ == "__main__":
    data = load_scam_data()
    embeddings = load_embeddings()
