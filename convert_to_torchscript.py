import torch
import torch.nn as nn
import torchvision.models as models
import os
from torch.serialization import safe_globals

# Định nghĩa class names
class_names = ["Benh loet", "Benh dom", "Khoe manh", "Benh ri sat", "Benh than thu"]

def load_model_directly(model_path):
    """Tải model trực tiếp từ file mà không cần khởi tạo kiến trúc"""
    print(f"Đang tải model từ {model_path}...")
    
    # Thêm EfficientNet vào safe_globals để tránh lỗi bảo mật
    from torchvision.models.efficientnet import EfficientNet
    
    # Tải model với weights_only=False
    try:
        with safe_globals([EfficientNet]):
            model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        print("Đã tải model thành công!")
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return None
    
    # Đặt model ở chế độ evaluation
    if hasattr(model, 'eval'):
        model.eval()
    
    return model

def create_wrapper_model(model):
    """Tạo một wrapper model để xử lý lỗi 'Could not get name of python class object'"""
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            
        def forward(self, x):
            return self.model(x)
    
    return ModelWrapper(model)

def convert_to_torchscript(model, output_path):
    """Chuyển đổi model sang TorchScript"""
    # Tạo input tensor mẫu
    example = torch.rand(1, 3, 224, 224)
    
    try:
        # Thử chuyển đổi trực tiếp
        traced_script_module = torch.jit.trace(model, example)
    except Exception as e:
        print(f"Lỗi khi trace model trực tiếp: {e}")
        print("Thử sử dụng wrapper model...")
        
        # Sử dụng wrapper model
        wrapped_model = create_wrapper_model(model)
        traced_script_module = torch.jit.trace(wrapped_model, example)
    
    # Lưu model
    traced_script_module.save(output_path)
    
    print(f"Đã lưu model TorchScript vào {output_path}")
    return output_path

def main():
    # Thư mục đầu ra
    output_dir = "app/src/main/assets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Đổi tên file model2.pt thành model.pt để dễ sử dụng
    old_path = os.path.join(output_dir, "model2.pt")
    new_path = os.path.join(output_dir, "model.pt")
    
    if os.path.exists(old_path):
        try:
            # Nếu file model.pt đã tồn tại, xóa nó
            if os.path.exists(new_path):
                os.remove(new_path)
            
            # Đổi tên file
            os.rename(old_path, new_path)
            print(f"Đã đổi tên {old_path} thành {new_path}")
        except Exception as e:
            print(f"Lỗi khi đổi tên file: {e}")
    else:
        print(f"Không tìm thấy file {old_path}")
    
    # Tạo file labels.txt
    labels_path = os.path.join(output_dir, "labels.txt")
    try:
        with open(labels_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(class_names))
        print(f"Đã tạo file {labels_path}")
    except Exception as e:
        print(f"Lỗi khi tạo file labels.txt: {e}")

if __name__ == "__main__":
    main() 