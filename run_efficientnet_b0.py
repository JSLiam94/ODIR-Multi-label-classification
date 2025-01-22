import os
os.environ["TORCH_NATIVE_USE_FLASH_ATTENTION"] = "0"
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from metrics_new import get_eval_metrics, print_metrics
from torchvision import transforms, models
import torch.optim as optim
from tqdm import tqdm
import logging
import timm
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torch.utils.tensorboard import SummaryWriter

# Set up logging
log_file = 'training_validation_metrics.log'
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((448, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print('data transform loaded')

# 自定义数据集类
class FundusDataset(Dataset):
    def __init__(self, anno_path, img_dir, transform=None):
        self.anno = pd.read_excel(anno_path)
        self.img_dir = img_dir
        self.transform = transform
        self.label_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
        self.num_classes = len(self.label_columns)

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        row = self.anno.iloc[idx]
        left_img_name = row['Left-Fundus']
        right_img_name = row['Right-Fundus']

        # 加载左眼图片
        left_img_path = os.path.join(self.img_dir, f"{left_img_name}")
        left_img = Image.open(left_img_path).convert('RGB')
        left_img = left_img.resize((224, 224))

        # 加载右眼图片
        right_img_path = os.path.join(self.img_dir, f"{right_img_name}")
        right_img = Image.open(right_img_path).convert('RGB')
        right_img = right_img.resize((224, 224))

        # 合并左右眼图片
        combined_img = Image.new('RGB', (left_img.width + right_img.width, left_img.height))
        combined_img.paste(left_img, (0, 0))
        combined_img.paste(right_img, (left_img.width, 0))

        if self.transform:
            combined_img = self.transform(combined_img)

        # 生成标签
        labels = torch.zeros(self.num_classes, dtype=torch.float32)  # 初始化标签向量
        for i, col in enumerate(self.label_columns):
            if row[col] == 1:
                labels[i] = 1

        return combined_img, labels

# 加载数据集
DATA_DIR = "D:/SUCAI/OIA-ODIR/"
train_anno_path = os.path.join(DATA_DIR, "Training-Set","Annotation","anno.xlsx")
train_img_dir = os.path.join(DATA_DIR, "Training-Set", "Images")

test_anno_path = os.path.join(DATA_DIR, "On-site","Annotation","anno.xlsx")
test_img_dir = os.path.join(DATA_DIR, "On-site", "Images")

train_dataset = FundusDataset(train_anno_path, train_img_dir, transform=data_transforms)
test_dataset = FundusDataset(test_anno_path, test_img_dir, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

print('dataset loaded')

# 加载EfficientNet模型
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('cuda loaded' + str(torch.cuda.is_available()))
model = model.to(device)
print('model loaded')

# 定义损失函数和优化器
criterion = torch.nn.BCEWithLogitsLoss()  # 多标签分类使用 BCEWithLogitsLoss
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
scaler = GradScaler()

# TensorBoard 日志记录
#writer = SummaryWriter()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_and_validate(model, train_loader, test_loader, criterion, optimizer, num_epochs=150, patience=7):
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        all_preds_train = []
        all_targets_train = []
        all_probs_train = []
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}')
        train_bar = tqdm(train_loader, desc='Training', leave=False)
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):  # 混合精度训练
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)  # 使用 sigmoid 将输出转换为概率
            preds = probs > 0.5  # 阈值设为 0.5
            all_preds_train.extend(preds.detach().cpu().numpy())
            all_targets_train.extend(labels.detach().cpu().numpy())
            all_probs_train.extend(probs.detach().cpu().numpy())
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()
            
            batch_f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
            train_bar.set_postfix(loss=running_loss / len(train_loader), F1=batch_f1)
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        
        # 验证阶段
        model.eval()
        running_loss_val = 0.0
        correct_val = 0
        total_val = 0
        all_preds_val = []
        all_targets_val = []
        all_probs_val = []

        val_bar = tqdm(test_loader, desc='Validating', leave=False)
        
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss_val += loss.item()
                probs = torch.sigmoid(outputs)
                preds = probs > 0.5
                all_probs_val.extend(probs.detach().cpu().numpy())
                all_preds_val.extend(preds.detach().cpu().numpy())
                all_targets_val.extend(labels.detach().cpu().numpy())
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()
                
                batch_f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
                val_bar.set_postfix(loss=running_loss_val / len(test_loader), F1=batch_f1)
        
        val_loss = running_loss_val / len(test_loader)
        val_accuracy = 100 * correct_val / total_val
        
        # 计算评估指标
        # 计算并输出指标
        train_metrics = get_eval_metrics(all_targets_train, all_preds_train)
        val_metrics = get_eval_metrics(all_targets_val, all_preds_val, probs_all=all_probs_val)
        
        # Log metrics
        logging.info(f'Epoch [{epoch+1}/{num_epochs}] - '
                     f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%\n'
                     f'Validation Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # Log training metrics per class
        logging.info("\n\nTraining Metrics (Overall):")
        for metric, value in train_metrics.items():
            if "report" not in metric:
                logging.info(f"{metric}: {value}")
        
        logging.info("Training Metrics (Per Class):")
        if "report" in train_metrics:
            class_report = train_metrics["report"]
            for class_label, metrics in class_report.items():
                if class_label.isdigit():  # 确保只处理具体类别的指标
                    logging.info(f"Class {class_label}: "
                                f"Precision: {metrics['precision']:.4f}, "
                                f"Recall: {metrics['recall']:.4f}, "
                                f"F1-score: {metrics['f1-score']:.4f}")
        
        # Log validation metrics per class
        logging.info("Validation Metrics (Overall):")
        for metric, value in val_metrics.items():
            if "report" not in metric:
                logging.info(f"{metric}: {value}")
        
        logging.info("Validation Metrics (Per Class):")
        if "report" in val_metrics:
            class_report = val_metrics["report"]
            for class_label, metrics in class_report.items():
                if class_label.isdigit():
                    logging.info(f"Class {class_label}: "
                                f"Precision: {metrics['precision']:.4f}, "
                                f"Recall: {metrics['recall']:.4f}, "
                                f"F1-score: {metrics['f1-score']:.4f}")
        
        # 学习率调度器
        scheduler.step(val_loss)
        
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 加载最佳模型权重
    model.load_state_dict(torch.load('checkpoint.pt'))
    #writer.close()

# 调用训练和验证过程
train_and_validate(model, train_loader, test_loader, criterion, optimizer)