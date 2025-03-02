# test_model.py
import torch
import torch.nn as nn

# Model cực kỳ đơn giản - chỉ là một lớp tuyến tính
class SuperSimpleModel(nn.Module):
    def __init__(self):
        super(SuperSimpleModel, self).__init__()
        self.fc = nn.Linear(3*224*224, 5)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc(x)

# Khởi tạo model
model = SuperSimpleModel()
model.eval()

# Input mẫu
example = torch.rand(1, 3, 224, 224)

# Xuất sang TorchScript
scripted_model = torch.jit.script(model)  # Thử script thay vì trace
scripted_model.save("app/src/main/assets/model.pt")

# Tạo file labels
with open("app/src/main/assets/labels.txt", "w") as f:
    f.write("Benh loet\nBenh dom\nKhoe manh\nBenh ri sat\nBenh than thu")

print("Đã tạo model đơn giản")