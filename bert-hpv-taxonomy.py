import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


batch_size =32 
num_epochs=4
lr=5e-5

print(f"batch_size: {batch_size:.4f}")
print(f"num_epochs: {num_epochs:.4f}")
print(f"Learning Rate: {lr:.4f}")


def load_json_file(path):
    with open(path, 'r') as file:
        return json.load(file)

train_data = load_json_file("train_labeled_hpv_with_taxonomies.json")['data']
dev_data = load_json_file("dev_labeled_hpv_with_taxonomies.json")['data']
test_data = load_json_file("test_labeled_hpv_with_taxonomies.json")['data']

taxonomy_list = ['misinformation', 'trust', 'civil_rights', 'literacy']

class HPV_Taxonomy_Dataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        taxonomies = item.get('taxonomies', [])
        label_vector = [1 if taxonomy in taxonomies else 0 for taxonomy in taxonomy_list]
        
        inputs = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label_vector, dtype=torch.float)
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BertForTaxonomyClassification(nn.Module):
    def __init__(self, n_taxonomies=len(taxonomy_list)):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_taxonomies)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        dropped = self.dropout(outputs.pooler_output)
        return torch.sigmoid(self.classifier(dropped))

train_dataset = HPV_Taxonomy_Dataset(train_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
dev_dataset = HPV_Taxonomy_Dataset(dev_data, tokenizer)
dev_loader = DataLoader(dev_dataset, batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForTaxonomyClassification().to(device)
optimizer = AdamW(model.parameters(), lr)


def train(model, train_loader, optimizer, num_epochs=4):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = nn.BCEWithLogitsLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

train(model, train_loader, optimizer)

def plot_taxonomy_distribution(taxonomy_list, predicted_counts, actual_counts, total_tweets, save_path='taxonomy_distribution.png'):
    predicted_percentages = [count / total_tweets * 100 for count in predicted_counts]
    actual_percentages = [count / total_tweets * 100 for count in actual_counts]

    x = range(len(taxonomy_list))  

    plt.figure(figsize=(10, 7))
    plt.bar(x, predicted_percentages, width=0.4, label='Predicted', align='center')
    plt.bar(x, actual_percentages, width=0.4, label='Actual', align='edge')
    plt.xlabel('Theme')
    plt.ylabel('Percentage of Tweets (%)')
    plt.title('Distribution of Tweets by Theme')
    plt.xticks(x, taxonomy_list, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()  

def evaluate(model, data_loader):
    model.eval()  
    true_labels = []
    predictions = []

    with torch.no_grad(): 
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()  
            output = model(input_ids, attention_mask)

            batch_predictions = (output > 0.5).int().cpu().numpy()
            predictions.extend(batch_predictions)
            true_labels.extend(labels)

    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    predicted_counts = predictions.sum(axis=0)
    actual_counts = true_labels.sum(axis=0)
    total_tweets = len(data_loader.dataset) 
    print('total_tweets:',total_tweets)   
    plot_taxonomy_distribution(taxonomy_list, predicted_counts, actual_counts, total_tweets, '/projects/klybarge/HPV_social_media/bert/hpv-taxonomy/taxonomy_distribution.png')
        
    return accuracy, precision, recall, f1


dev_accuracy, dev_precision, dev_recall, dev_f1 = evaluate(model, dev_loader)
print("\nDev Set Evaluation Metrics:")
print(f"Accuracy: {dev_accuracy:.4f}")
print(f"Precision: {dev_precision:.4f}")
print(f"Recall: {dev_recall:.4f}")
print(f"F1 Score: {dev_f1:.4f}")

test_dataset = HPV_Taxonomy_Dataset(test_data, tokenizer)
test_loader = DataLoader(test_dataset, batch_size)
test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader)
print("\nTest Set Evaluation Metrics:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")