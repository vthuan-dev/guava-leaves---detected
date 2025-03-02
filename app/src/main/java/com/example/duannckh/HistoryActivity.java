package com.example.duannckh;

import android.content.Intent;
import android.database.Cursor;
import android.os.Bundle;
import android.util.Log; // Import Log để debug
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ListView;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import java.util.ArrayList;

public class HistoryActivity extends AppCompatActivity {

    private static final String TAG = "HistoryActivity"; // Tag để dễ debug
    private static final String TAG2 = "HEHE";
    private ListView listViewHistory;
    private Button btnBack, btnViewHistoryDetail, btnClearHistory, btnClearAllHistory;
    private DatabaseHelper dbHelper;
    private HistoryItem selectedItem;
    private int selectedPosition = -1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_history);

        Log.d(TAG, "onCreate: Khởi tạo giao diện lịch sử"); // Log khi vào Activity

        listViewHistory = findViewById(R.id.listView_history);
        btnBack = findViewById(R.id.btn_back);
        btnViewHistoryDetail = findViewById(R.id.btn_view_history_detail);
        btnClearHistory = findViewById(R.id.btn_clear_history);
        btnClearAllHistory = findViewById(R.id.btn_clear_all_history);
        dbHelper = new DatabaseHelper(this);

        int userId = getIntent().getIntExtra("userId", -1);
        Log.d(TAG, "onCreate: userId nhận được = " + userId);

        if (userId == -1) {
            Toast.makeText(this, "Lỗi: Không tìm thấy ID người dùng", Toast.LENGTH_SHORT).show();
            finish(); // Đóng activity nếu không có userId hợp lệ
            return;
        }

        loadHistoryData(userId);

        listViewHistory.setOnItemClickListener((parent, view, position, id) -> {
            selectedItem = (HistoryItem) parent.getItemAtPosition(position);
            selectedPosition = position;

            Log.d(TAG, "Mục đã chọn: " + selectedItem.getImageName());

            view.setBackgroundColor(getResources().getColor(android.R.color.holo_blue_light));
            ((HistoryAdapter) listViewHistory.getAdapter()).setSelectedPosition(position);
            ((HistoryAdapter) listViewHistory.getAdapter()).notifyDataSetChanged();
        });

        btnBack.setOnClickListener(v -> {
            Log.d(TAG, "Nhấn nút quay lại");
            Intent intent = new Intent(HistoryActivity.this, HomeActivity.class);
            intent.putExtra("userId", userId);
            startActivity(intent);
            finish();
        });

        btnViewHistoryDetail.setOnClickListener(v -> {
            if (selectedItem != null) {
                Log.d(TAG, "Mở chi tiết lịch sử: " + selectedItem.getImageName());
                Intent intent = new Intent(HistoryActivity.this, HistoryDetailActivity.class);
                intent.putExtra("imageName", selectedItem.getImageName());
                intent.putExtra("dateTime", selectedItem.getDateTime());
                intent.putExtra("diseaseName", selectedItem.getDiseaseName());

                intent.putExtra("userId", userId);
                startActivity(intent);
            } else {
                Toast.makeText(this, "Vui lòng chọn một mục trong lịch sử", Toast.LENGTH_SHORT).show();
                Log.d(TAG, "Không có mục nào được chọn");
            }
        });

        btnClearHistory.setOnClickListener(v -> {
            if (selectedItem != null) {
                String imageName = selectedItem.getImageName();
                Log.d(TAG, "Xóa mục lịch sử: " + imageName);

                dbHelper.deleteHistoryByImageName(imageName);
                loadHistoryData(userId);

                selectedItem = null;
                selectedPosition = -1;

                Toast.makeText(this, "Đã xóa mục đã chọn", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Vui lòng chọn một mục để xóa", Toast.LENGTH_SHORT).show();
                Log.d(TAG, "Không có mục nào được chọn để xóa");
            }
        });

        btnClearAllHistory.setOnClickListener(v -> {
            Log.d(TAG, "Nhấn nút xóa toàn bộ lịch sử");
            new AlertDialog.Builder(HistoryActivity.this)
                    .setTitle("Xác nhận xóa")
                    .setMessage("Bạn có chắc chắn muốn xóa toàn bộ danh sách lịch sử của bạn?")
                    .setPositiveButton("Có", (dialog, which) -> {
                        dbHelper.clearUserHistory(userId);
                        loadHistoryData(userId);
                        Toast.makeText(this, "Đã xóa toàn bộ lịch sử của bạn", Toast.LENGTH_SHORT).show();
                        Log.d(TAG, "Đã xóa toàn bộ lịch sử của userId: " + userId);
                    })
                    .setNegativeButton("Không", (dialog, which) -> dialog.dismiss())
                    .show();
        });
    }

    private void loadHistoryData(int userId) {
        Log.d(TAG2, "Tải dữ liệu lịch sử cho userId: " + userId);
        ArrayList<HistoryItem> historyList = new ArrayList<>();
        Cursor cursor = dbHelper.getAllHistory();

        if (cursor != null && cursor.moveToFirst()) {
            do {
                int recordUserId = cursor.getInt(cursor.getColumnIndexOrThrow("id"));
                Log.d(TAG2, String.valueOf(recordUserId));
                if (recordUserId == userId) {
                    String imageName = cursor.getString(cursor.getColumnIndexOrThrow("image_name"));
                    String dateTime = cursor.getString(cursor.getColumnIndexOrThrow("date_time"));
                    String diseaseName = cursor.getString(cursor.getColumnIndexOrThrow("disease"));
                    String medicineName = cursor.getString(cursor.getColumnIndexOrThrow("treatment_drug"));
                    String medicineImage = cursor.getString(cursor.getColumnIndexOrThrow("image_medicine"));

                    Log.d(TAG2, "Lịch sử: " + imageName + " - " + dateTime + " - " + diseaseName + " - " + medicineName + " - " + medicineImage);
                    HistoryItem historyItem = new HistoryItem(imageName, dateTime, diseaseName,medicineName, medicineImage);
                    historyList.add(historyItem);
                }
            } while (cursor.moveToNext());
            Log.d(TAG2, "HUHU");

            cursor.close();
        } else {
            Log.d(TAG2, "Không có dữ liệu lịch sử");
        }
        Log.d(TAG2, "Số phần tử trong historyList: " + historyList.size());
        HistoryAdapter adapter = new HistoryAdapter(this, R.layout.history_item, historyList);
        listViewHistory.setAdapter(adapter);

    }

}
