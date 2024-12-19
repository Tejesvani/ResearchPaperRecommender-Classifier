#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset

ds = load_dataset("gfissore/arxiv-abstracts-2021")


# In[2]:


print(ds)


# In[3]:


# Convert to DataFrame
df = ds['train'].to_pandas()


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.columns


# In[7]:


df.isnull().sum()


# In[11]:


# Select 25% of the data randomly
df_downsampled = df.sample(frac=0.001, random_state=42)

# Check the shape of the down-sampled dataset
print(df_downsampled.shape)


# In[12]:


# Select only the necessary columns for the task
df_downsampled = df_downsampled[['id', 'title', 'abstract', 'categories']]

df_downsampled.isnull().sum()


# In[37]:


# Import libraries for data manipulation and visualization
import pandas as pd
import re

# Clean text data (remove special characters, extra spaces, etc.)
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text.strip()

df_downsampled['title'] = df_downsampled['title'].apply(clean_text)
df_downsampled['abstract'] = df_downsampled['abstract'].apply(clean_text)

# Display cleaned data
print(df_downsampled.head(10))


# In[14]:


# Combine title and abstract for feature extraction
df_downsampled['combined_text'] = df_downsampled['title'] + " " + df_downsampled['abstract']


# In[15]:


df_downsampled['combined_text'] = df_downsampled['combined_text'].astype(str)
# df_downsampled['categories'] = df_downsampled['categories'].astype(str)


# In[16]:


df_downsampled.info()


# In[17]:


df_downsampled['categories'] = df_downsampled['categories'].apply(lambda x: str(x))


# In[18]:


from sklearn.preprocessing import LabelEncoder


# In[19]:


label_encoder = LabelEncoder()
df_downsampled['categories_encoded'] = label_encoder.fit_transform(df_downsampled['categories'])


# In[20]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df_downsampled['combined_text'],
    df_downsampled['categories_encoded'],
    test_size=0.05,
    random_state=42
)


# In[21]:


print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)


# In[22]:


X_train


# In[23]:


from sentence_transformers import SentenceTransformer
import torch

# Load SentenceTransformer and specify the device
model_name = "all-MiniLM-L6-v2"
sentence_model = SentenceTransformer(model_name, device='cuda')
#sentence_model.half()  # Convert model to half precision

# Set batch size (adjust based on your GPU memory)
batch_size = 64

# Vectorize text data
with torch.no_grad():
    X_train_embeddings = sentence_model.encode(
        X_train.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        device='cuda'
    )
    X_test_embeddings = sentence_model.encode(
        X_test.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        device='cuda'
    )

print("Vectorization complete!")


# ## Vectorization

# ## Paper Recommender

# In[24]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define a function to recommend papers
def recommend_papers(query_embedding, train_embeddings, df_train, top_k=5):

    # Compute cosine similarities between the query and the training embeddings
    similarities = cosine_similarity([query_embedding], train_embeddings)[0]

    # Get indices of top K most similar papers
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Prepare the recommendations
    recommendations = []
    for idx in top_indices:
        paper_id = df_train.iloc[idx]['id']
        title = df_train.iloc[idx]['title']
        score = similarities[idx]
        recommendations.append((paper_id, title, score))

    return recommendations

# Example: Use the first paper in the test set as the query
query_index = 0  # Adjust as needed
query_embedding = X_test_embeddings[query_index]

# Get recommendations
recommendations = recommend_papers(query_embedding, X_train_embeddings, df_downsampled)

# Convert recommendations to a DataFrame
recommendations_df = pd.DataFrame(recommendations, columns=["ID", "Title", "Similarity"])

# Print the DataFrame
print("Top recommendations:")
recommendations_df


# ## Subject Area Prediction

# In[25]:


print("X_train.head")
print(X_train.head())
print("X_test.head")
print(X_test.head())

print("y_train.head")
print(y_train.head())
print("y_test.head")
print(y_test.head())


# In[26]:


print("X_train_embeddings.shape")
print(X_train_embeddings.shape)
print("X_test_embeddings.shape")
print(X_test_embeddings.shape)
print("y_train.shape")
print(y_train.shape)
print("y_test.shape")
print(y_test.shape)


# In[27]:


df_downsampled.info()


# In[29]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to predict top 3 subject areas for a new research paper based on similarity
def predict_top_subject_areas(new_paper: str, X_train_embeddings, y_train, sentence_model):
    # Step 1: Vectorize the new paper text
    new_paper_vectorized = sentence_model.encode([new_paper], show_progress_bar=True)

    # Step 2: Calculate cosine similarity between the new paper and all training samples
    similarities = cosine_similarity(new_paper_vectorized, X_train_embeddings)

    # Step 3: Get top 3 most similar entries
    top_3_indices = np.argsort(similarities[0])[-3:][::-1]  # Sorting and getting top 3 indices

    # Step 4: Get corresponding subject areas for top 3 indices
    top_3_labels = y_train.iloc[top_3_indices]
    top_3_scores = similarities[0][top_3_indices]

    # Step 5: Map the encoded labels back to actual categories
    top_3_subject_areas = label_encoder.inverse_transform(top_3_labels)

    # Step 6: Save results to a DataFrame
    result_df = pd.DataFrame({
        "Rank": range(1, 4),  # Rank from 1 to 3
        "Subject Area": top_3_subject_areas,
        "Similarity Score": top_3_scores
    })

    return result_df

# Example usage: Predict top 3 subject areas for a new research paper
new_paper = X_test.iloc[1]  # Get first entry from the test data
result_df = predict_top_subject_areas(new_paper, X_train_embeddings, y_train, sentence_model)

# Display the DataFrame
print("\nTop 3 Predicted Subject Areas with Scores as DataFrame:")
print(result_df)


# In[30]:


from ast import literal_eval

# getting unique labels
labels_column = result_df['Subject Area'].apply(literal_eval)
labels = labels_column.explode().unique()

#Splitting each category by spaces and creating a set of unique categories
unique_categories = set(cat for label in labels for cat in label.split())

# Convert the set back to a sorted list for better readability
unique_categories_list = sorted(unique_categories)

# Display the results
print("Categories:", unique_categories_list)


# In[27]:


label_encoder = LabelEncoder()
all_labels = np.concatenate([y_train, y_test])
label_encoder.fit(all_labels)
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# In[28]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Vectorize text data
embeddings = sentence_model.encode(df_downsampled['combined_text'].tolist(), show_progress_bar=True)

# Prepare labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df_downsampled['categories'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_labels, test_size=0.2, random_state=42)

# Create PyTorch datasets
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the neural network model
class SubjectAreaPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SubjectAreaPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Set up the model
input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = len(label_encoder.classes_)
model = SubjectAreaPredictor(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy:.4f}")

# Function to predict subject areas for new papers
def predict_subject_areas(new_paper_text, top_k=3):
    model.eval()
    new_paper_embedding = sentence_model.encode([new_paper_text])
    with torch.no_grad():
        output = model(torch.FloatTensor(new_paper_embedding).to(device))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k)

    top_k_subject_areas = label_encoder.inverse_transform(top_k_indices.cpu().numpy()[0])
    return list(zip(top_k_subject_areas, top_k_probs.cpu().numpy()[0]))

# Example usage
new_paper_text = "This paper discusses the effects of dark matter on galaxy formation."
predictions = predict_subject_areas(new_paper_text)
print("Top 3 predicted subject areas:")
for subject, probability in predictions:
    print(f"{subject}: {probability:.4f}")


# In[32]:


# Save the trained model
torch.save(model.state_dict(), "subject_area_predictor.pth")

# Save the label encoder classes
np.save("label_classes.npy", label_encoder.classes_)

print("Model and label encoder saved!")


# In[30]:


# Define the model class again (must match the saved model's structure)
class SubjectAreaPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SubjectAreaPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Load the label encoder
label_classes = np.load("label_classes.npy", allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes

# Define the model structure and load the saved weights
input_dim = 384  # Adjust based on your sentence embedding size
hidden_dim = 128
output_dim = len(label_encoder.classes_)
model = SubjectAreaPredictor(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("subject_area_predictor.pth"))
model.eval()  # Set model to evaluation mode

print("Model and label encoder loaded!")


# In[36]:


import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import gc

# Set device and clear cache
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Reduce batch size
batch_size = 2

# Create data loaders (assuming you have train_dataset and val_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Load pre-trained DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_encoder.classes_))
model.to(device)

# Set up optimizer and gradient scaler
optimizer = AdamW(model.parameters(), lr=2e-5)
scaler = GradScaler()

# Training loop with gradient accumulation and mixed precision
num_epochs = 3
accumulation_steps = 4

for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Clear cache after each step
        torch.cuda.empty_cache()
        gc.collect()

    # Validation loop (implement as needed)

# Function to predict subject areas (implement as needed)


# In[31]:


import ipywidgets as widgets
from IPython.display import display

# Function to predict subject areas for new papers
def predict_subject_areas(new_paper_text, top_k=3):
    model.eval()
    new_paper_embedding = sentence_model.encode([new_paper_text])
    with torch.no_grad():
        output = model(torch.FloatTensor(new_paper_embedding))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k)

    top_k_subject_areas = label_encoder.inverse_transform(top_k_indices.cpu().numpy()[0])
    return list(zip(top_k_subject_areas, top_k_probs.cpu().numpy()[0]))

# Define widgets
input_box = widgets.Textarea(
    description="Paper Text:",
    placeholder="Enter the text of the research paper here...",
    layout=widgets.Layout(width="500px", height="100px")
)
output_label = widgets.Label(value="Predicted Subject Areas: ")
button = widgets.Button(description="Predict", button_style="success")

# Define prediction function for UI
def on_button_click(b):
    paper_text = input_box.value
    if paper_text.strip():
        predictions = predict_subject_areas(paper_text, top_k=3)
        output = "\n".join([f"{subject}: {probability:.4f}" for subject, probability in predictions])
        output_label.value = f"Predicted Subject Areas:\n{output}"
    else:
        output_label.value = "Please enter a valid paper text."

# Attach function to button click
button.on_click(on_button_click)

# Display UI
display(input_box, button, output_label)

