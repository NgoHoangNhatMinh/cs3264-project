#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ## Data pre-processing

# In[3]:


def preprocess_casas(file_path):
    # Read the CSV without parsing dates yet
    df = pd.read_csv(
        file_path, 
        sep=',', 
        names=['Date', 'Time', 'Sensor', 'Status', 'Activity'],
        engine='c' # Faster engine
    )

    # Manually combine Date and Time into a single datetime column
    # We join strings first, then convert. 
    df['Date_Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # Pivot so sensors are columns
    df_pivot = df.pivot_table(index='Date_Time', columns='Sensor', values='Status', aggfunc='last')

    # Standardize statuses to numeric 1/0
    # Map common CASAS statuses to bits
    status_map = {'ON': 1, 'OFF': 0, 'OPEN': 1, 'CLOSE': 0}
    df_numeric = df_pivot.replace(status_map)

    # Force convert any remaining strings to NaN, then fill
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

    # Resample to 1-minute intervals and forward-fill
    # This ensures the timeline is continuous for the Transformer
    df_resampled = df_numeric.resample('1min').last().ffill().fillna(0)

    return df_resampled

data = preprocess_casas('../data/labeled/hh101.csv')
print(f"Data shape: {data.shape}")
print(data.head())


# In[4]:


num_sensors = data.shape[1]
print(f"Your model will need an input_dim of: {num_sensors}")


# ## Dataset

# In[5]:


class ProjectDataset(Dataset):
    def __init__(self, dataframe, window_size=60):
        self.data = torch.FloatTensor(dataframe.values.copy())
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # Extract a window of 60 minutes
        return self.data[idx : idx + self.window_size]

# Usage
train_ds = ProjectDataset(data) # 'data' is your preprocessed DataFrame
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)


# ## Core Model

# In[6]:


class ProjectTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        # 1. Linear Projection (Numeric Pseudo-Embedding)
        self.input_projection = nn.Linear(input_dim, d_model)

        # 2. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Decoder Head (Reconstruction)
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_projection(x)
        latent = self.transformer_encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


# In[7]:


def detect_anomaly(model, sequence, threshold):
    model.eval()
    sequence = sequence.to(device)

    with torch.no_grad():
        reconstruction = model(sequence)
        # Calculate reconstruction loss per time step
        loss = torch.mean((reconstruction - sequence) ** 2, dim=-1)

        # If loss > threshold, alert!
        is_anomaly = loss > threshold
        return is_anomaly, loss


# ### Utils

# In[8]:


def save_checkpoint(model, optimizer, epoch, loss, filename="../checkpoints/project_checkpoint.pth"):
    checkpoint_dir = os.path.dirname(filename)

    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created directory: {checkpoint_dir}")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, filename="../checkpoints/project_checkpoint.pth"):
    if os.path.exists(filename):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filename, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch
    else:
        print("No checkpoint found, starting from scratch.")
        return 0


# ## Training

# In[9]:


def train_model(model, dataloader, epochs=10):
    start_epoch = load_checkpoint(model, optimizer)

    if start_epoch >= epochs:
        print(f"Model already trained for {start_epoch} epochs. Increase 'epochs' to train further.")
        return

    model.to(device)
    model.train()

    for epoch in range(start_epoch, epochs):
        total_loss = 0

        for batch in dataloader:
            # batch shape: [batch_size, seq_len, input_dim]
            batct = batch.to(device)

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        save_checkpoint(model, optimizer, epoch, avg_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")


# In[10]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = data.shape[1] 
model = ProjectTransformer(input_dim=input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss().to(device)


# In[11]:


train_model(model, train_loader)


# ## Calculating Normalcy Threshold

# In[12]:


def calculate_threshold(model, dataloader):
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in dataloader:
            output = model(batch)
            # Calculate MSE for each individual window in the batch
            mse = torch.mean((output - batch)**2, dim=(1, 2))
            errors.extend(mse.tolist())

    errors = np.array(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    # Threshold = Mean + 3 * Standard Deviation
    threshold = mean_error + (3 * std_error)

    print(f"Mean Normal Error: {mean_error:.6f}")
    print(f"Suggested Threshold: {threshold:.6f}")
    return threshold

# threshold = calculate_threshold(model, val_loader)

