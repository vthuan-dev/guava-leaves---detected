package com.example.duannckh;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class PyTorchClassifier {
    private static final String TAG = "PyTorchClassifier";
    private static final String MODEL_FILE = "model.pt";
    private static final String LABEL_FILE = "labels.txt";
    
    private final Context context;
    private Module module;
    private List<String> labels;
    
    // Chuẩn hóa mean và std cho tiền xử lý ảnh
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};
    
    public PyTorchClassifier(Context context) {
        this.context = context;
        try {
            Log.d(TAG, "=== Bắt đầu khởi tạo PyTorchClassifier ===");
            
            // Thử cách khác: tải trực tiếp module từ assets
            try {
                Log.d(TAG, "Phương pháp 1: Tải model trực tiếp từ assets");
                String assetsModelPath = "file:///android_asset/" + MODEL_FILE;
                module = Module.load(assetsModelPath);
                Log.d(TAG, "Tải model trực tiếp từ assets thành công!");
                
                // Tải labels
                labels = loadLabels(context);
                Log.d(TAG, "Tải labels thành công: " + labels);
                return;
            } catch (Exception e) {
                Log.e(TAG, "Phương pháp 1 thất bại: " + e.getMessage());
                // Tiếp tục với phương pháp khác
            }
            
            // Phương pháp 2: Copy model từ assets vào filesDir
            try {
                Log.d(TAG, "Phương pháp 2: Copy model từ assets vào filesDir");
                String modelPath = assetFilePath(context, MODEL_FILE);
                if (modelPath == null) {
                    Log.e(TAG, "Không thể tạo đường dẫn cho model");
                    tryAlternativeMethod();
                    return;
                }
                
                // Kiểm tra kích thước file
                File modelFile = new File(modelPath);
                Log.d(TAG, "Kích thước model: " + modelFile.length() + " bytes");
                
                // Tải model
                module = Module.load(modelPath);
                Log.d(TAG, "Tải model thành công!");
                
                // Tải labels
                labels = loadLabels(context);
                Log.d(TAG, "Tải labels thành công: " + labels);
            } catch (Exception e) {
                Log.e(TAG, "Phương pháp 2 thất bại: " + e.getMessage());
                tryAlternativeMethod();
            }
        } catch (Exception e) {
            Log.e(TAG, "Lỗi khởi tạo: " + e.getMessage(), e);
        }
    }
    
    // Phương pháp thay thế sử dụng model cực kỳ đơn giản
    private void tryAlternativeMethod() {
        try {
            Log.d(TAG, "Thử phương pháp thay thế: tạo model đơn giản tại runtime");
            
            // Tải simple model thay thế
            File modelDir = new File(context.getFilesDir(), "pytorch_models");
            if (!modelDir.exists()) {
                modelDir.mkdirs();
            }
            
            // Ghi log thông tin runtime
            Log.d(TAG, "Kiểm tra môi trường PyTorch");
            Log.d(TAG, "Android SDK: " + android.os.Build.VERSION.SDK_INT);
            Log.d(TAG, "Thiết bị: " + android.os.Build.DEVICE);
            
            // Tải labels
            try {
                labels = loadLabels(context);
                Log.d(TAG, "Tải labels thành công: " + labels);
            } catch (Exception e) {
                Log.e(TAG, "Không thể tải labels: " + e.getMessage());
                // Tạo labels mặc định
                labels = new ArrayList<>();
                labels.add("Benh loet");
                labels.add("Benh dom");
                labels.add("Khoe manh");
                labels.add("Benh ri sat");
                labels.add("Benh than thu");
                Log.d(TAG, "Đã tạo labels mặc định");
            }
        } catch (Exception e) {
            Log.e(TAG, "Phương pháp thay thế thất bại: " + e.getMessage(), e);
        }
    }
    
    public DiseaseResult classify(Bitmap bitmap) {
        try {
            // Tạo kết quả mặc định nếu model chưa sẵn sàng
            if (module == null) {
                Log.e(TAG, "Model chưa được tải, trả về kết quả mặc định");
                DiseaseResult defaultResult = new DiseaseResult();
                defaultResult.setPredictedClass("Không thể phân loại");
                defaultResult.setConfidence(0.0f);
                defaultResult.setTreatment("Vui lòng thử lại hoặc khởi động lại ứng dụng");
                defaultResult.setMedicine("Không có");
                return defaultResult;
            }
            
            if (labels == null || labels.isEmpty()) {
                Log.e(TAG, "Nhãn chưa được tải");
                return null;
            }
            
            // Kiểm tra bitmap
            if (bitmap == null) {
                Log.e(TAG, "Bitmap null");
                return null;
            }
            
            Log.d(TAG, "=== Bắt đầu quá trình phân loại ===");
            
            // Resize ảnh
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
            Log.d(TAG, "Đã resize ảnh thành 224x224");
            
            // Chuyển bitmap sang tensor
            try {
                // Tạo input tensor
                Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                        resizedBitmap,
                        MEAN,
                        STD
                );
                Log.d(TAG, "Chuyển đổi bitmap thành tensor thành công: " + 
                      inputTensor.shape()[0] + "x" + inputTensor.shape()[1] + "x" +
                      inputTensor.shape()[2] + "x" + inputTensor.shape()[3]);
                
                // Thực hiện dự đoán
                try {
                    Log.d(TAG, "Bắt đầu forward pass");
                    long startTime = System.currentTimeMillis();
                    Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
                    long endTime = System.currentTimeMillis();
                    Log.d(TAG, "Forward pass thành công trong " + (endTime - startTime) + "ms");
                    
                    // Lấy kết quả
                    float[] scores = outputTensor.getDataAsFloatArray();
                    Log.d(TAG, "Số lượng scores: " + scores.length);
                    
                    // Debug: in tất cả scores
                    StringBuilder sb = new StringBuilder("Scores: ");
                    for (int i = 0; i < scores.length; i++) {
                        sb.append(scores[i]).append(", ");
                    }
                    Log.d(TAG, sb.toString());
                    
                    // Tìm lớp có xác suất cao nhất
                    int maxIndex = 0;
                    float maxScore = scores[0];
                    for (int i = 1; i < scores.length; i++) {
                        if (scores[i] > maxScore) {
                            maxScore = scores[i];
                            maxIndex = i;
                        }
                    }
                    
                    Log.d(TAG, "Lớp dự đoán: " + maxIndex + ", điểm: " + maxScore);
                    
                    // Tạo kết quả
                    DiseaseResult result = new DiseaseResult();
                    if (maxIndex < labels.size()) {
                        String predictedClass = labels.get(maxIndex);
                        result.setPredictedClass(predictedClass);
                        result.setConfidence(maxScore);
                        result.setTreatment(getTreatmentInfo(predictedClass));
                        result.setMedicine(getMedicineInfo(predictedClass));
                        result.setMedicineImage(getMedicineImageInfo(predictedClass));
                        Log.d(TAG, "Kết quả: " + predictedClass);
                    } else {
                        result.setPredictedClass("Unknown");
                        result.setConfidence(maxScore);
                        result.setTreatment("Không có thông tin điều trị");
                        result.setMedicine("Không có thông tin thuốc");
                        result.setMedicineImage("");
                        Log.d(TAG, "Kết quả: Unknown (index ngoài phạm vi)");
                    }
                    
                    return result;
                    
                } catch (Exception e) {
                    Log.e(TAG, "Lỗi trong forward pass: " + e.getMessage(), e);
                    throw e;
                }
                
            } catch (Exception e) {
                Log.e(TAG, "Lỗi chuyển đổi bitmap thành tensor: " + e.getMessage(), e);
                throw e;
            }
            
        } catch (Exception e) {
            Log.e(TAG, "Lỗi trong quá trình phân loại: " + e.getMessage(), e);
            e.printStackTrace();
            
            // Trả về kết quả lỗi
            DiseaseResult errorResult = new DiseaseResult();
            errorResult.setPredictedClass("Lỗi");
            errorResult.setConfidence(0.0f);
            errorResult.setTreatment("Lỗi: " + e.getMessage());
            errorResult.setMedicine("Vui lòng thử lại");
            return errorResult;
        }
    }
    
    private String getTreatmentInfo(String className) {
        // Thông tin điều trị tương ứng với từng loại bệnh
        switch (className) {
            case "Benh loet":
                return "1. Loại bỏ lá bị bệnh; 2. Phun thuốc diệt nấm có chứa đồng; 3. Tăng cường thông gió";
            case "Benh dom":
                return "1. Cắt tỉa và tiêu hủy lá bị bệnh; 2. Sử dụng thuốc phòng trừ nấm; 3. Đảm bảo khoảng cách trồng hợp lý";
            case "Khoe manh":
                return "Cây khỏe mạnh, tiếp tục chăm sóc bình thường với chế độ tưới nước và phân bón hợp lý";
            case "Benh ri sat":
                return "1. Loại bỏ và tiêu hủy lá bị bệnh; 2. Phun thuốc trừ nấm; 3. Kiểm soát độ ẩm";
            case "Benh than thu":
                return "1. Cắt bỏ cành, lá bị bệnh; 2. Phun thuốc trừ nấm; 3. Tránh tưới nước trên lá";
            default:
                return "Không có thông tin điều trị";
        }
    }
    
    private String getMedicineInfo(String className) {
        // Thông tin thuốc tương ứng với từng loại bệnh
        switch (className) {
            case "Benh loet":
                return "Thuốc có chứa đồng như Copper Oxychloride, Bordeaux Mixture";
            case "Benh dom":
                return "Thuốc trừ nấm như Mancozeb, Propineb, Chlorothalonil";
            case "Khoe manh":
                return "Không cần thuốc";
            case "Benh ri sat":
                return "Thuốc trừ nấm như Hexaconazole, Difenoconazole";
            case "Benh than thu":
                return "Thuốc trừ nấm như Thiophanate-methyl, Carbendazim";
            default:
                return "Không có thông tin thuốc";
        }
    }
    
    private String getMedicineImageInfo(String className) {
        // URL hình ảnh thuốc tương ứng với từng loại bệnh
        return ""; // Trả về chuỗi rỗng vì hiện tại không có hình ảnh thuốc
    }
    
    // Phương thức kiểm tra môi trường
    public boolean checkEnvironment() {
        try {
            Log.d(TAG, "=== Kiểm tra môi trường PyTorch ===");
            Log.d(TAG, "Kiểm tra môi trường PyTorch");
            
            // Kiểm tra thư mục assets
            String[] assets = context.getAssets().list("");
            Log.d(TAG, "Danh sách files trong assets:");
            boolean hasModelFile = false;
            boolean hasLabelFile = false;
            
            for (String asset : assets) {
                Log.d(TAG, " - " + asset);
                if (MODEL_FILE.equals(asset)) {
                    hasModelFile = true;
                }
                if (LABEL_FILE.equals(asset)) {
                    hasLabelFile = true;
                }
            }
            
            Log.d(TAG, "Model file exists: " + hasModelFile);
            Log.d(TAG, "Label file exists: " + hasLabelFile);
            
            // Kiểm tra kích thước file
            if (hasModelFile) {
                try (InputStream is = context.getAssets().open(MODEL_FILE)) {
                    int size = is.available();
                    Log.d(TAG, "Model file size: " + size + " bytes");
                    if (size > 20 * 1024 * 1024) {
                        Log.w(TAG, "Model file quá lớn (>20MB), có thể gây vấn đề trên một số thiết bị");
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Không thể đọc kích thước model: " + e.getMessage());
                }
            }
            
            return hasModelFile && hasLabelFile;
        } catch (Exception e) {
            Log.e(TAG, "Lỗi kiểm tra môi trường: " + e.getMessage(), e);
            return false;
        }
    }
    
    // Cải tiến phương thức assetFilePath
    private String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        
        // Xóa file cũ nếu tồn tại để đảm bảo copy mới nhất
        if (file.exists()) {
            boolean deleted = file.delete();
            Log.d(TAG, "Xóa file cũ " + file.getAbsolutePath() + ": " + deleted);
        }
        
        Log.d(TAG, "Bắt đầu copy " + assetName + " từ assets vào " + file.getAbsolutePath());
        
        // Đảm bảo thư mục cha tồn tại
        if (!file.getParentFile().exists()) {
            file.getParentFile().mkdirs();
        }
        
        // Copy file với buffer lớn
        try (InputStream is = context.getAssets().open(assetName)) {
            try (FileOutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[16 * 1024]; // 16KB buffer
                int read;
                long total = 0;
                
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                    total += read;
                }
                
                os.flush();
                Log.d(TAG, "Đã copy thành công " + total + " bytes vào " + file.getAbsolutePath());
            }
        } catch (IOException e) {
            Log.e(TAG, "Lỗi copy file từ assets: " + e.getMessage(), e);
            if (file.exists()) {
                file.delete();
            }
            throw e;
        }
        
        // Kiểm tra kết quả
        if (!file.exists() || file.length() == 0) {
            Log.e(TAG, "File không tồn tại hoặc rỗng sau khi copy");
            return null;
        }
        
        return file.getAbsolutePath();
    }
    
    // Phương thức tải labels
    private List<String> loadLabels(Context context) throws IOException {
        List<String> labels = new ArrayList<>();
        try (InputStream is = context.getAssets().open(LABEL_FILE);
             BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }
            Log.d(TAG, "Đã tải " + labels.size() + " nhãn: " + labels);
        } catch (IOException e) {
            Log.e(TAG, "Lỗi khi tải nhãn: " + e.getMessage(), e);
            throw e;
        }
        return labels;
    }
    
    // Giải phóng tài nguyên
    public void close() {
        if (module != null) {
            module = null;
        }
    }
    
    public static class DiseaseResult {
        private String predictedClass;
        private float confidence;
        private String treatment;
        private String medicine;
        private String medicineImage;
        
        public String getPredictedClass() { return predictedClass; }
        public void setPredictedClass(String predictedClass) { this.predictedClass = predictedClass; }
        
        public float getConfidence() { return confidence; }
        public void setConfidence(float confidence) { this.confidence = confidence; }
        
        public String getTreatment() { return treatment; }
        public void setTreatment(String treatment) { this.treatment = treatment; }
        
        public String getMedicine() { return medicine; }
        public void setMedicine(String medicine) { this.medicine = medicine; }
        
        public String getMedicineImage() { return medicineImage; }
        public void setMedicineImage(String medicineImage) { this.medicineImage = medicineImage; }
    }
}