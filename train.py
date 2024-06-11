import os
import torch
from models.transformers import create_cnn_model
from utils.data_model import load_data_pytorch
import matplotlib.pyplot as plt

# Parameters
img_size = (200, 200)
input_shape = (3, 200, 200)
num_classes = 3
train_dir = "D:/Kuliah/Semester 6/Pengenalan Pola/Code/animal_detection/cell/train/"
val_dir = "D:/Kuliah/Semester 6/Pengenalan Pola/Code/animal_detection/cell/validation/"

# Load data
train_loader, val_loader, class_names = load_data_pytorch(train_dir, val_dir, img_size)

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_cnn_model(input_shape, num_classes).to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=40):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")
    
    return train_losses

# Train the model
train_losses = train_model(model, train_loader, criterion, optimizer)

# Save the model
save_dir = 'models'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'my_model.pth')
torch.save(model.state_dict(), save_path)

# Plot training loss
plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Grafik Loss')
plt.legend()
plt.show()
print(plt.show())