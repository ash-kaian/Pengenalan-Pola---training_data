import torch
from models.transformers import create_cnn_model
from utils.data_model import load_data_pytorch

# Parameters
img_size = (200, 200)
input_shape = (3, 200, 200)
num_classes = 3 
train_dir = "D:/Kuliah/Semester 6/Pengenalan Pola/Code/animal_detection/cell/train/"
val_dir = "D:/Kuliah/Semester 6/Pengenalan Pola/Code/animal_detection/cell/validation/"

# Load data
_, val_loader, class_names = load_data_pytorch(train_dir, val_dir, img_size)

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_cnn_model(input_shape, num_classes).to(device)

# Load the model
model_path = 'models/my_model.pth'
model.load_state_dict(torch.load(model_path))
model.to(device)

# Evaluation function
def evaluate_model(model, val_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    print(f'Test sample: {total_samples}')
    print(f'Test correct: {total_correct}')
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Evaluate the model
evaluate_model(model, val_loader)