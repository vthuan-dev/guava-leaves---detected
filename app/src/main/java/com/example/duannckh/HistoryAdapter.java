package com.example.duannckh;

import android.content.Context;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.io.File;
import java.util.ArrayList;

public class HistoryAdapter extends ArrayAdapter<HistoryItem> {

    private Context context;
    private int resource;
    private int selectedPosition = -1;
    private static final String TAG = "HistoryAdapter";

    public HistoryAdapter(@NonNull Context context, int resource, @NonNull ArrayList<HistoryItem> objects) {
        super(context, resource, objects);
        this.context = context;
        this.resource = resource;
    }

    @NonNull
    @Override
    public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
        if (convertView == null) {
            LayoutInflater inflater = LayoutInflater.from(context);
            convertView = inflater.inflate(resource, parent, false);
        }

        // Lấy item hiện tại
        HistoryItem historyItem = getItem(position);
        if (historyItem != null) {
            Log.d(TAG, "getView: Item tại vị trí " + position + " - " + historyItem.getImageName() + ", " + historyItem.getDateTime() + ", " + historyItem.getDiseaseName());
        } else {
            Log.d(TAG, "getView: Item tại vị trí " + position + " là null");
        }

        // Tham chiếu các view trong item_history.xml
        ImageView imageView = convertView.findViewById(R.id.image_icon);
        TextView textViewNgayGio = convertView.findViewById(R.id.text_date_time);
        TextView disease = convertView.findViewById(R.id.text_disease_name);

        // Hiển thị dữ liệu
        if (historyItem != null) {
            // Hiển thị hình ảnh từ đường dẫn trong HistoryItem
            File imgFile = new File(context.getFilesDir(), historyItem.getImageName());
            if (imgFile.exists()) {
                Log.d(TAG, "getView: Tìm thấy tệp hình ảnh: " + imgFile.getAbsolutePath());
                Bitmap bitmap = BitmapFactory.decodeFile(imgFile.getAbsolutePath());
                imageView.setImageBitmap(bitmap);
            } else {
                Log.d(TAG, "getView: Không tìm thấy tệp hình ảnh: " + imgFile.getAbsolutePath());
            }

            disease.setText(historyItem.getDiseaseName());
            Log.d(TAG, "getView: Disease: " + historyItem.getDiseaseName());

            textViewNgayGio.setText(historyItem.getDateTime());
            Log.d(TAG, "getView: DateTime: " + historyItem.getDateTime());

            if (position == selectedPosition) {
                Log.d(TAG, "getView: Vị trí " + position + " được chọn, thay đổi màu nền");
                convertView.setBackgroundColor(getContext().getResources().getColor(android.R.color.holo_blue_light));
            } else {
                Log.d(TAG, "getView: Vị trí " + position + " không được chọn, màu nền giữ nguyên");
                convertView.setBackgroundColor(getContext().getResources().getColor(android.R.color.transparent));
            }
        }
        return convertView;
    }

    public void setSelectedPosition(int position) {
        Log.d(TAG, "setSelectedPosition: Đặt vị trí đã chọn: " + position);
        selectedPosition = position;
    }
}
