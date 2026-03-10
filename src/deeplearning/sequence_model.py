import torch
import torch.nn as nn
import logging
import os
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

class SessionRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=1):
        super(SessionRNN, self).__init__()
        # Lớp Embedding: Biểu diễn từng Item ID thành một vector dense d-chiều
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        
        # Mạng GRU (Gated Recurrent Unit): Tốt hơn RNN thuần túy trong việc nhớ chuỗi dài
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Lớp phân loại: Ánh xạ vector hidden cuối cùng sang xác suất của toàn bộ từ vựng (Next-item prediction)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        """
        x: tensor(batch_size, sequence_length) chứa các Item Index
        """
        embedded = self.embedding(x) # (batch_size, seq_len, emb_dim)
        
        # Chạy RNN/GRU qua chuỗi
        output, hidden = self.gru(embedded)
        
        # Chỉ lấy hidden state của bước thời gian cuối cùng để dự đoán
        # output[:, -1, :] shape: (batch_size, hidden_dim)
        last_hidden = output[:, -1, :]
        
        # logits cho tất cả các class (items)
        logits = self.fc(last_hidden)
        return logits


class SequenceRecommender:
    def __init__(self, model_path=None):
        self.model = None
        self.item2idx = {"<PAD>": 0, "<UNK>": 1} # 0 for padding, 1 for unknown
        self.idx2item = {0: "<PAD>", 1: "<UNK>"}
        self.item_metadata = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def predict_next(self, current_session, top_n=5):
        """
        Inference: Nhận một chuỗi các Item_IDs (Giỏ hàng hiện tại)
        Dự đoán Top-N sản phẩm sẽ được bỏ vào giỏ hàng tiếp theo
        """
        if not self.model:
            logging.warning("Mô hình chưa được nạp.")
            return []
            
        self.model.eval()
        
        # Chuyển đổi Item ID -> Index
        indices = [self.item2idx.get(item, self.item2idx["<UNK>"]) for item in current_session]
        
        # Chuẩn bị Tensor (1 batch)
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # Suy luận forward()
            logits = self.model(input_tensor)
            
            # Tính xác suất bằng Softmax
            probabilities = torch.softmax(logits, dim=1).squeeze()
            
            # Lấy top N giá trị dự đoán có xác suất cao nhất
            top_probs, top_indices = torch.topk(probabilities, top_n + 1) # Lấy n+1 để phòng trường hợp predict trúng <PAD> hoac <UNK>
            
        recommendations = []
        for prob, idx in zip(top_probs, top_indices):
            idx_val = idx.item()
            prob_val = prob.item()
            
            # Bỏ qua Padding và Unknown
            if idx_val in (0, 1):
                continue
                
            item_id = self.idx2item[idx_val]
            
            # Lọc bỏ những sản phẩm đã có trong giỏ hàng hiện tại (để khỏi gợi ý lại chính nó)
            if item_id in current_session:
                continue
                
            meta = self.item_metadata.get(item_id, {"id": item_id, "title": f"Sản phẩm {item_id}", "image": "https://via.placeholder.com/150"})
            rec = meta.copy()
            rec['confidence'] = round(prob_val * 100, 2) # Percent
            recommendations.append(rec)
            
            if len(recommendations) >= top_n:
                break
                
        return recommendations
