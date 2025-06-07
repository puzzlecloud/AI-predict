import os
import torch
import torch.nn as nn
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import csv
import numpy as np

# ==== 配置 ====
class Config:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    mode = 'train'
    log_path = 'mynewmodel.pth'
    image_size = 64
    batch_size = 8
    cnn_epoch = 50
    mlp_epoch = 20000
    cnn_lr = 0.0005
    mlp_lr = 0.001

cfg = Config()

# ==== 数据集类 ====
class MyDataset(Dataset):
    def __init__(self, train=True, transforms=None):
        self.train = train
        self.transforms = transforms
        data_center = pd.read_csv(os.path.join(cfg.root, 'result_new1300km.csv'))
        #data_center = pd.read_csv(os.path.join(cfg.root, 'result_new100.csv'))
        samples = []
        grouped = data_center.groupby("Id")
        for tid, group in grouped:
            group = group.sort_values(by='TIME')
            if len(group) < 11: continue
            for i in range(len(group) - 10):
                seq = group.iloc[i:i+11]
                input_paths = [f"{seq.iloc[j]['Id']}.{seq.iloc[j]['TIME']}" for j in range(10)]
                label_path = f"{seq.iloc[10]['Id']}.{seq.iloc[10]['TIME']}"
                center_pre = (seq.iloc[9]['LAT'], seq.iloc[9]['LONG'])
                center_cur = (seq.iloc[10]['LAT'], seq.iloc[10]['LONG'])
                samples.append((input_paths, label_path, center_pre, center_cur))

        split = int(0.8 * len(samples))
        self.samples = samples[:split] if train else samples[split:]

    def __getitem__(self, index):
        input_paths, label_path, center_pre, center_cur = self.samples[index]
        images = []
        time_steps = []

        for i, path in enumerate(input_paths):
            tid, _ = path.split(".")
            img_dir = os.path.join(cfg.root, 'pic_2014-2022_1300km', tid)
            img_stack = []
            for var in ['mwd', 'mwp', 'swh', 'v10']:
                p = os.path.join(img_dir, var, f"{path}.{var}.png")
                img = cv2.imread(p)
                img = cv2.resize(img, (cfg.image_size, cfg.image_size))
                img_stack.append(img)
            img = cv2.merge(img_stack)
            img_tensor = self.transforms(img) if self.transforms else torch.from_numpy(img.transpose(2, 0, 1)).float() / 255
            images.append(img_tensor)
            time_steps.append(i)

        input_tensor = torch.stack(images)       # (10, 12, H, W)
        time_tensor = torch.tensor(time_steps)   # (10,)

        tid, _ = label_path.split(".")
        label_dir = os.path.join(cfg.root, 'pic_2014-2022_1300km', tid)
        label_stack = []
        for var in ['mwd', 'mwp', 'swh', 'v10']:
            p = os.path.join(label_dir, var, f"{label_path}.{var}.png")
            img = cv2.imread(p)
            img = cv2.resize(img, (cfg.image_size, cfg.image_size))
            label_stack.append(img)
        label_img = cv2.merge(label_stack)
        label_tensor = self.transforms(label_img) if self.transforms else torch.from_numpy(label_img.transpose(2, 0, 1)).float() / 255

        return input_tensor, time_tensor, label_tensor, label_path, center_pre, center_cur

    def __len__(self):
        return len(self.samples)


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(10, embed_dim)

    def forward(self, t):  # (B, T)
        return self.embedding(t)  # (B, T, D)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):  # (B, T, D)
        return self.transformer(x)

class TimeTransformerNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=12, embed_dim=32, image_size=64):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.image_size = image_size

        # 初始通道嵌入
        self.spatial_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.time_embed = TimeEmbedding(embed_dim)

        # Transformer 编码器
        self.transformer = TransformerEncoder(d_model=embed_dim, nhead=4, num_layers=2)

        # 解码器结构
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, t):  # x: (B, T, C, H, W), t: (B, T)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.spatial_embed(x)  # (B*T, D, H, W)
        x = x.view(B, T, self.embed_dim, H, W).permute(0, 3, 4, 1, 2)  # (B, H, W, T, D)

        time_emb = self.time_embed(t)  # (B, T, D)
        x = x + time_emb.view(B, 1, 1, T, self.embed_dim)  # 时间位置编码相加

        x = x.permute(0, 1, 2, 4, 3).contiguous().view(B * H * W, T, self.embed_dim)  # (B*H*W, T, D)
        x = self.transformer(x)  # (B*H*W, T, D)
        x = x[:, -1, :].view(B, H, W, self.embed_dim).permute(0, 3, 1, 2)  # (B, D, H, W)

        out = self.decoder(x)  # (B, C, H, W)
        return out


# ==== MLP 模块 ====
class CenterMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=2, dropout_prob=1e-6):
        super(CenterMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout_prob),  
            nn.Linear(hidden_dim, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim)  
        )
    
    def forward(self, x):
        return self.model(x)




def detect_center_from_image(img_rgb):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    # Ensure the image is of type CV_8UC1 (8-bit single channel)
    gray = np.uint8(gray)  # Convert the float32 to uint8

    # Threshold the image to create a binary image
    _, binary = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    all_contours = np.concatenate(contours)
    rect = cv2.minAreaRect(all_contours)
    return rect[0]
    


def pixel_to_latlon(x, y, pre_lat, pre_lon):
    # 计算纬度
    lat = pre_lat + (y - cfg.image_size // 2) * (11.8 / cfg.image_size)
    
    # 计算经度的跨度，纬度影响经度的转换
    delta_lon = (2 * 1300 / (111 * np.cos(lat * np.pi / 180))) / cfg.image_size
    
    # 计算经度
    lon = pre_lon - (x - cfg.image_size // 2) * delta_lon
    
    # 返回精确到小数点后两位的经纬度
    return round(lat, 2), round(lon, 2)



def geodistance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(lambda x: x * np.pi / 180.0, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return round(2 * np.arcsin(np.sqrt(a)) * 6371, 3)
        


def train_cnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MyDataset(train=True, transforms=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = TimeTransformerNet(in_channels=12, out_channels=12).to(device)
    optimizer_cnn = torch.optim.Adam(model.parameters(), lr=cfg.cnn_lr)
    loss_fn = nn.MSELoss()

    os.makedirs('debug_pred_images', exist_ok=True)
    variable_names = ['mwd', 'mwp', 'swh', 'v10']

    csv_path = 'features_targets.csv'
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Open CSV file for writing once at the start (with the header)
    with open(csv_path, 'w') as f:
        f.write('mwd_x,mwd_y,mwp_x,mwp_y,swh_x,swh_y,v10_x,v10_y,center_pre_lat,center_pre_lon,center_cur_lat,center_cur_lon,loss\n')

    for epoch in range(cfg.cnn_epoch):
        model.train()
        total_loss = 0

        for i, (x, t, y, _, center_pre, center_cur) in enumerate(loader):
            x = x.to(device)
            t = t.to(device)
            y = y.to(device)

            # Forward pass
            pred = model(x, t)
            loss = loss_fn(pred, y)

            optimizer_cnn.zero_grad()
            loss.backward()
            optimizer_cnn.step()

            total_loss += loss.item()

            # Iterate through the batch and process each sample
            for j in range(cfg.batch_size):  # for each sample in the batch
                # Ensure index j is within bounds
                if j >= len(center_pre[0]) or j >= len(center_cur[0]):
                    continue  # Skip if index j is out of bounds

                # Extract the predicted and true images for the current sample
                pred_img = pred[j].detach().cpu().numpy().transpose(1, 2, 0)
                true_img = y[j].detach().cpu().numpy().transpose(1, 2, 0)

                # Ensure the pixel values are within valid range (0, 255)
                pred_img = np.clip(pred_img * 255, 0, 255).astype(np.uint8)
                true_img = np.clip(true_img * 255, 0, 255).astype(np.uint8)

                # Save the predicted and true images
                for var in variable_names:
                    pred_var = pred_img[:, :, variable_names.index(var)*3:(variable_names.index(var)+1)*3]
                    true_var = true_img[:, :, variable_names.index(var)*3:(variable_names.index(var)+1)*3]

                    # Save the image
                    cv2.imwrite(f'debug_pred_images/epoch{epoch + 1}_batch{j+1}_{var}_pred.png', pred_var)
                    cv2.imwrite(f'debug_pred_images/epoch{epoch + 1}_batch{j+1}_{var}_true.png', true_var)

                # Extract the center coordinates from the predicted images
                single_feature = []
                valid = True
                for k in range(4):
                    sub_img = pred_img[:, :, k * 3:(k + 1) * 3]
                    center = detect_center_from_image(sub_img)
                    if center is not None:
                        single_feature.extend(center)
                    else:
                        valid = False
                        break

                # If valid features, write the data to CSV
                if valid:
                    pre_lat, pre_lon = center_pre[0][j].item(), center_pre[1][j].item()
                    cur_lat, cur_lon = center_cur[0][j].item(), center_cur[1][j].item()

                    # Ensure single_feature has exactly 8 values before writing to CSV
                    if len(single_feature) == 8:
                        with open(csv_path, 'a') as f:
                            line = ','.join(map(str, single_feature)) + f',{pre_lat},{pre_lon},{cur_lat},{cur_lon},{loss.item():.4f}\n'
                            f.write(line)
                            print(f'Successfully wrote row for batch {j}')

        # Print average loss for the current epoch
        print(f"📖 Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")
        torch.save(model.state_dict(), cfg.log_path)



def train_mlp():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取CSV文件
    csv_path = 'features_targets.csv'
    df = pd.read_csv(csv_path, header=0)

    # 提取X和Y
    X = df.iloc[:, :10].values  # 前10个特征作为X
    Y = df.iloc[:, 10:12].values  # 最后两个值是标签，作为Y  

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

    # 初始化MLP模型
    mlp = CenterMLP(input_dim=10).to(device)
    optimizer_mlp = optim.Adam(mlp.parameters(), lr=cfg.mlp_lr)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')  # 初始化最小loss为正无穷

    # 创建并打开 CSV 文件用于记录训练损失
    with open('mlp_training_loss.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])  

        # 训练MLP模型
        mlp.train()
        for epoch in range(cfg.mlp_epoch):
            optimizer_mlp.zero_grad()

            pred_coords = mlp(X_tensor)
            loss = loss_fn(pred_coords, Y_tensor)

            loss.backward()
            optimizer_mlp.step()

            current_loss = loss.item()
            writer.writerow([epoch + 1, current_loss])
            print(f"Epoch {epoch + 1}/{cfg.mlp_epoch}, Loss: {current_loss:.4f}")

            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(mlp.state_dict(), 'mlp_best_model.pth')
                print(f"✅ Saved best model at epoch {epoch + 1} with loss {best_loss:.4f}")

            if epoch == cfg.mlp_epoch - 1:
                pred_vals = pred_coords[:10].detach().cpu().numpy()
                true_vals = Y_tensor[:10].detach().cpu().numpy()
                for i in range(10):
                    pred_lat, pred_lon = pred_vals[i]
                    true_lat, true_lon = true_vals[i]
                    print(f"Pred: ({pred_lat:.2f}, {pred_lon:.2f}) | True: ({true_lat:.2f}, {true_lon:.2f})")

    print("Training complete. Best model saved as mlp_best_model.pth.")



def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载 TimeTransformerNet 模型
    cnn_model = TimeTransformerNet(in_channels=12, out_channels=12).to(device)
    cnn_model.load_state_dict(torch.load(cfg.log_path, map_location=device))
    cnn_model.eval()

    # 加载 CenterMLP 模型
    mlp_model = CenterMLP().to(device)
    mlp_model.load_state_dict(torch.load('mlp_best_model.pth', map_location=device))
    mlp_model.eval()

    # 加载测试数据集
    dataset = MyDataset(train=False, transforms=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_pred_coords = []
    all_true_coords = []
    all_mae_km = []  # 用于存储所有 MAE（公里）
    all_squared_errors = []  # 用于存储每个样本的误差平方

    # 打开 CSV 文件以记录 MAE 和 RMSE
    with open('test_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pred_Lat', 'Pred_Lon', 'True_Lat', 'True_Lon', 'MAE_km', 'RMSE_km'])  # 写入表头

        for input_tensor, time_tensor, _, name, pre, cur in loader:
            input_tensor = input_tensor.to(device)  # (1, T, C, H, W)
            time_tensor = time_tensor.to(device)    # (1, T)

            # 使用 CNN 模型进行图像预测
            with torch.no_grad():
                pred = cnn_model(input_tensor, time_tensor)[0].cpu().numpy().transpose(1, 2, 0)
                pred = (pred * 255).clip(0, 255).astype(np.uint8)

            features = []
            valid = True
            pre_lat, pre_lon = float(pre[0]), float(pre[1])
            true_lat, true_lon = float(cur[0]), float(cur[1])

            # 从预测图像中提取每个变量的中心坐标
            for i, var in enumerate(['mwd', 'mwp', 'swh', 'v10']):
                img = pred[:, :, i * 3:(i + 1) * 3]
                c = detect_center_from_image(img)
                if c:
                    x_pix, y_pix = c
                    features.extend([x_pix, y_pix])
                else:
                    valid = False
                    break

            if valid:
                input_tensor_mlp = torch.tensor([features + [pre_lat, pre_lon]], dtype=torch.float32).to(device)
                with torch.no_grad():
                    pred_coord = mlp_model(input_tensor_mlp)[0].detach().cpu().numpy()
                    pred_lat, pred_lon = pred_coord

                    all_pred_coords.append(pred_coord)
                    all_true_coords.append([true_lat, true_lon])

                    error = geodistance(pred_lon, pred_lat, true_lon, true_lat)

                    # 计算 MAE（公里）：误差的绝对值
                    mae_km = error  # MAE 是误差的绝对值，单位为公里

                    # 计算每个样本的误差的平方，并累加
                    squared_error = error ** 2
                    all_squared_errors.append(squared_error)

                    # 将预测值和真实值写入 CSV 文件
                    writer.writerow([pred_lat, pred_lon, true_lat, true_lon, mae_km, np.sqrt(np.mean(all_squared_errors))])

                    # 保存 MAE 和 RMSE
                    all_mae_km.append(mae_km)

                    print(f"Pred: ({pred_lat:.2f}, {pred_lon:.2f}) | True: ({true_lat:.2f}, {true_lon:.2f}) | MAE: {mae_km:.2f} km")

   
    if all_squared_errors:
        avg_mae = np.mean(all_mae_km)
        rmse_km = np.sqrt(np.mean(all_squared_errors))  
        print(f"\n 平均 MAE 距离: {avg_mae:.2f} km")
        print(f" 平均 RMSE 距离：{rmse_km:.2f} km")


if __name__ == '__main__':
    if cfg.mode == 'train':
        #train_cnn()
        #train_mlp()
        test()
    elif cfg.mode == 'test':
        test()
