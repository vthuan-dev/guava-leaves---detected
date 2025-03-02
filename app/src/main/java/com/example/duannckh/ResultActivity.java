package com.example.duannckh;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

import java.io.File;

public class ResultActivity extends AppCompatActivity {
    private static final String TAG = "ResultActivity";
    
    private ImageView imageView;
    private TextView tvPrediction, tvTreatment, tvMedicine;
    private Button btnBack, btnViewInfo;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);
        
        // Ánh xạ view - sử dụng đúng ID từ layout
        imageView = findViewById(R.id.imageView);
        tvPrediction = findViewById(R.id.tvPrediction);  // Giả sử đây là ID đúng
        tvTreatment = findViewById(R.id.tvTreatment);
        tvMedicine = findViewById(R.id.tvMedicine);
        btnBack = findViewById(R.id.btnBack);
        btnViewInfo = findViewById(R.id.btnViewInfo);  // Giả sử đây là ID đúng
        
        // Lấy dữ liệu từ ResultData singleton
        String prediction = ResultData.getInstance().getPrediction();
        String confidence = ResultData.getInstance().getConfidence();
        String treatment = ResultData.getInstance().getTreatment();
        String medicine = ResultData.getInstance().getMedicine();
        String imagePath = ResultData.getInstance().getImagePath();
        
        Log.d(TAG, "Dữ liệu nhận được: prediction=" + prediction + 
              ", confidence=" + confidence +
              ", treatment=" + treatment + 
              ", medicine=" + medicine + 
              ", imagePath=" + imagePath);
        
        // Hiển thị kết quả
        if (prediction != null && !prediction.isEmpty()) {
            String displayText = prediction;
            if (confidence != null && !confidence.isEmpty()) {
                displayText += " (" + confidence + ")";
            }
            tvPrediction.setText(displayText);
        } else {
            tvPrediction.setText("Unknown");
        }
        
        // Hiển thị cách trị
        if (treatment != null && !treatment.isEmpty()) {
            tvTreatment.setText(treatment);
        } else {
            tvTreatment.setText("Không có thông tin");
        }
        
        // Hiển thị thuốc
        if (medicine != null && !medicine.isEmpty()) {
            tvMedicine.setText(medicine);
        } else {
            tvMedicine.setText("Không có thông tin");
        }
        
        // Hiển thị ảnh
        if (imagePath != null && !imagePath.isEmpty()) {
            File imgFile = new File(imagePath);
            if (imgFile.exists()) {
                Bitmap bitmap = BitmapFactory.decodeFile(imgFile.getAbsolutePath());
                imageView.setImageBitmap(bitmap);
            }
        }
        
        // Xử lý sự kiện nút Back
        btnBack.setOnClickListener(v -> {
            finish();
        });
        
        // Xử lý sự kiện nút Info - Tạm thời chỉ hiển thị thông báo
        if (btnViewInfo != null) {
            btnViewInfo.setOnClickListener(v -> {
                // Thay vì chuyển đến InfoActivity (chưa tồn tại)
                // Chỉ hiển thị thông báo đơn giản
                android.widget.Toast.makeText(ResultActivity.this, 
                    "Thông tin chi tiết về: " + prediction, 
                    android.widget.Toast.LENGTH_LONG).show();
                
                // Nếu muốn thêm InfoActivity sau này, bạn có thể mở comment dòng dưới
                // Intent intent = new Intent(ResultActivity.this, InfoActivity.class);
                // intent.putExtra("disease", prediction);
                // startActivity(intent);
            });
        }
    }
}
