package com.example.duannckh;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;

public class HistoryDetailActivity extends AppCompatActivity {

    private static final String TAG = "HistoryDetailActivity";

    private TextView TextNgayGioDetail, Textresult, Texttreament, TextMedicineInfo;
    private ImageView imageViewHistoryDetail,imageMedicine;
    private Button btnBackToMain, btnback;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_history_detail);

        Log.d(TAG, "onCreate: Bắt đầu khởi tạo giao diện.");

        // Tham chiếu các view
        TextNgayGioDetail = findViewById(R.id.tv_time_info);
        Textresult = findViewById(R.id.tv_result_info);
        Texttreament = findViewById(R.id.tv_treatment_info);
        TextMedicineInfo = findViewById(R.id.tv_medicine_info);
        imageViewHistoryDetail = findViewById(R.id.image_result);
        btnBackToMain = findViewById(R.id.btn_home);
        btnback = findViewById(R.id.btn_back);
        imageMedicine = findViewById(R.id.image_medicine);

        Log.d(TAG, "onCreate: Đã tham chiếu tất cả các view.");

        int userId = getIntent().getIntExtra("userId", -1);
        Log.d(TAG, "onCreate: userId nhận được từ Intent là: " + userId);

        // Nhận dữ liệu từ Intent
        String imageName = getIntent().getStringExtra("imageName");
        String ngayGio = getIntent().getStringExtra("dateTime");
        String disease = getIntent().getStringExtra("diseaseName");

        Log.d(TAG, "onCreate: Dữ liệu nhận được từ Intent - imageName: " + imageName + ", dateTime: " + ngayGio + ", diseaseName: " + disease);

        if (imageName != null && ngayGio != null && disease != null) {
            // Hiển thị thời gian
            TextNgayGioDetail.setText(ngayGio);

            DatabaseHelper dbHelper = new DatabaseHelper(this);
            Log.d(TAG, "onCreate: Đã khởi tạo DatabaseHelper.");

            Cursor cursor = dbHelper.getHistoryByImageName(imageName);
            Log.d(TAG, "onCreate: Đã thực thi truy vấn lấy lịch sử với imageName: " + imageName);

            String image_medicine = null;
            if (cursor != null && cursor.moveToFirst()) {
                Log.d(TAG, "onCreate: Cursor không rỗng, bắt đầu lấy dữ liệu từ cơ sở dữ liệu.");
                String treatment = cursor.getString(cursor.getColumnIndexOrThrow(DatabaseHelper.COLUMN_TREATMENT));
                String treatmentDrug = cursor.getString(cursor.getColumnIndexOrThrow(DatabaseHelper.COLUMN_TREATMENT_DRUG));
                image_medicine = cursor.getString(cursor.getColumnIndexOrThrow(DatabaseHelper.COLUMN_IMAGE_NAME_MEDICINE));
                // Hiển thị thông tin trên các TextView
                Textresult.setText(disease);
                Texttreament.setText(treatment);
                TextMedicineInfo.setText(treatmentDrug);

                Log.d(TAG, "onCreate: Dữ liệu lấy được - treatment: " + treatment + ", treatmentDrug: " + treatmentDrug);

                cursor.close();
                Log.d(TAG, "onCreate: Cursor đã được đóng.");
            } else {
                Log.e(TAG, "onCreate: Không tìm thấy dữ liệu trong cơ sở dữ liệu.");
                Toast.makeText(this, "Không tìm thấy dữ liệu trong lịch sử", Toast.LENGTH_SHORT).show();
            }

            // Kiểm tra hình ảnh từ drawable theo tên hình ảnh
            int imageResId = getResources().getIdentifier(image_medicine, "drawable", getPackageName());
            if (imageResId != 0) {
                // Nếu hình ảnh tồn tại trong drawable, hiển thị nó
                imageMedicine.setImageResource(imageResId);
                Log.d(TAG, "onCreate: Đã tìm thấy hình ảnh trong drawable: " + imageName);
            } else {
                // Nếu không tìm thấy, hiển thị thông báo lỗi
                Log.e(TAG, "onCreate: Không tìm thấy hình ảnh trong drawable: " + imageName);
                Toast.makeText(this, "Không tìm thấy hình ảnh trong drawable", Toast.LENGTH_SHORT).show();
            }
            File imgFile = new File(getFilesDir(), imageName);
            if (imgFile.exists()) {
                Bitmap bitmap = BitmapFactory.decodeFile(imgFile.getAbsolutePath());
                imageViewHistoryDetail.setImageBitmap(bitmap);
            } else {
                Toast.makeText(this, "Không tìm thấy hình ảnh", Toast.LENGTH_SHORT).show();
            }
        } else {
            Log.e(TAG, "onCreate: Dữ liệu nhận từ Intent không hợp lệ.");
            Toast.makeText(this, "Dữ liệu không hợp lệ", Toast.LENGTH_SHORT).show();
        }

        // Xử lý sự kiện nút quay lại
        btnBackToMain.setOnClickListener(v -> {
            Log.d(TAG, "onCreate: Người dùng nhấn nút Quay về trang chủ.");
            Intent intent = new Intent(HistoryDetailActivity.this, HomeActivity.class);
            intent.putExtra("userId", userId);
            startActivity(intent);
            finish();
        });

        btnback.setOnClickListener(v -> {
            Log.d(TAG, "onCreate: Người dùng nhấn nút Quay lại danh sách lịch sử.");
            Intent intent = new Intent(HistoryDetailActivity.this, HistoryActivity.class);
            intent.putExtra("userId", userId);
            startActivity(intent);
            finish();
        });
    }
}