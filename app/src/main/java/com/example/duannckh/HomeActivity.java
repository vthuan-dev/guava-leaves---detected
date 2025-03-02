package com.example.duannckh;

import android.annotation.SuppressLint;
import android.app.Dialog;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Point;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Display;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import androidx.viewpager2.widget.ViewPager2;

import android.database.sqlite.SQLiteDatabase;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class HomeActivity extends AppCompatActivity {
    private ViewPager2 viewPager;
    private Handler handler = new Handler(Looper.getMainLooper());
    private int currentPage = 0;
    private CarouselAdapter adapter;
    private List<View> indicators = new ArrayList<>();
    private static final String API_KEY ="7a1d56a0bbecbb8ff074cc44ddf4fbd6" ;
    private ImageButton btnViewDisease, btnViewHistory, btnHome;
    private TextView diseaseName, diseaseDescription,textViewNhietDo ,textViewKhaNangMua,  textViewDoAm;
    private DatabaseHelper dbHelper;
   // private Handler handler = new Handler();
    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        try {
            setContentView(R.layout.activity_home);

            // Khởi tạo các nút và TextView
            btnViewDisease = findViewById(R.id.btn_view_disease);
            btnViewHistory = findViewById(R.id.btn_view_history);
            btnHome = findViewById(R.id.btn_home);
           // diseaseName = findViewById(R.id.disease_name);
            //diseaseDescription = findViewById(R.id.disease_description);
            textViewNhietDo = findViewById(R.id.temperature_value);
            textViewKhaNangMua = findViewById(R.id.rain_chance_value);
            textViewDoAm = findViewById(R.id.humidity_value);
            dbHelper = new DatabaseHelper(this);
            String city="Cần Thơ";//tteentinhr sẽ được hiển thị thông tin về nhiệt độ
            layDuLieuThoiTiet(city);
            int userId = getIntent().getIntExtra("userId", -1);
            Toast.makeText(HomeActivity.this, "Đăng nhập thành công. ID: " + userId, Toast.LENGTH_SHORT).show();
            viewPager = findViewById(R.id.viewPager);
            List<CarouselItem> items = new ArrayList<>();
            items.add(new CarouselItem(R.drawable.hinhnen_home1, "Bệnh thán thư"));
            items.add(new CarouselItem(R.drawable.hinhnen_home2, "Bệnh thối rễ"));
            items.add(new CarouselItem(R.drawable.hinhnen_home3, "Cây khỏe"));
            adapter = new CarouselAdapter(items, (view, position) -> {
                String content = "";
                String content2 = "";
                String content3 = "";


                switch (position) {
                    case 0:
                        content = "Bệnh thán thư";
                        content2 = "hinhnen_home1";
                        content3 = "Bệnh thán thư trên cây ổi do nấm Colletotrichum gloeosporioides gây ra, thường xuất hiện trong điều kiện thời tiết nóng ẩm, đặc biệt vào mùa mưa. Triệu chứng bệnh biểu hiện rõ trên lá, quả và cành...";
                        break;
                    case 1:
                        content = "Bệnh thối rễ";
                        content2 = "hinhnen_home2";
                        content3 = "Bệnh thối rễ trên cây ổi thường xuất hiện khi cây bị nhiễm nấm, vi khuẩn hoặc các tác nhân gây hại khác...";
                        break;
                    case 2:
                        content = "Cây khỏe";
                        content2 = "hinhnen_home3";
                        content3 = "Cây không có dấu hiệu bệnh, phát triển tốt và khỏe mạnh.";
                        break;
                }


                showPopup(view, content, content2, content3);
            });
            viewPager.setAdapter(adapter);
            setupIndicators(adapter.getItemCount());


            viewPager.registerOnPageChangeCallback(new ViewPager2.OnPageChangeCallback() {
                @Override
                public void onPageSelected(int position) {
                    super.onPageSelected(position);
                    updateIndicators(position);
                }
            });


            startAutoScroll();










            // Xử lý khi nhấn nút "Xem bệnh" để chuyển sang MainActivity
            btnViewDisease.setOnClickListener(v -> {
                Intent intent = new Intent(HomeActivity.this, MainActivity.class);
                intent.putExtra("userId", userId);
                startActivity(intent);
            });

            // Xử lý khi nhấn nút "Xem lịch sử"
            btnViewHistory.setOnClickListener(v -> {
                Intent intent = new Intent(HomeActivity.this, HistoryActivity.class);
                intent.putExtra("userId", userId);
                startActivity(intent);
            });

            // Xử lý khi nhấn nút "home" để reset lại trang hiện tại
            btnHome.setOnClickListener(v -> {
                Intent intent = new Intent(HomeActivity.this, HomeActivity.class);
                intent.putExtra("userId", userId);
                startActivity(intent);
                finish(); // Kết thúc HomeActivity hiện tại để reset lại trang
            });
        } catch (Exception e) {
            Toast.makeText(this, "Đã xảy ra lỗi trong quá trình khởi tạo.", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
    }
    private void startAutoScroll() {
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                if (viewPager.getAdapter() != null) {
                    currentPage = (currentPage == viewPager.getAdapter().getItemCount() - 1) ? 0 : currentPage + 1;
                    viewPager.setCurrentItem(currentPage, true);
                    handler.postDelayed(this, 5000);
                }
            }
        }, 5000);
    }


    private void setupIndicators(int count) {
        LinearLayout indicatorLayout = findViewById(R.id.indicatorLayout);
        indicatorLayout.removeAllViews();
        indicators.clear();


        for (int i = 0; i < count; i++) {
            View view = new View(this);
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(20, 20);
            params.setMargins(15, 0, 15, 0);
            view.setLayoutParams(params);
            view.setBackgroundResource(R.drawable.indicator_unselected);
            indicatorLayout.addView(view);
            indicators.add(view);
        }


        updateIndicators(0);
    }


    private void updateIndicators(int position) {
        for (int i = 0; i < indicators.size(); i++) {
            View view = indicators.get(i);
            LinearLayout.LayoutParams params = (LinearLayout.LayoutParams) view.getLayoutParams();
            if (i == position) {
                params.width = 25;
                params.height = 25;
                view.setBackgroundResource(R.drawable.indicator_selected);
            } else {
                params.width = 20;
                params.height = 20;
                view.setBackgroundResource(R.drawable.indicator_unselected);
            }
            view.setLayoutParams(params);
        }
    }


    private void showPopup(View view, String content, String content2, String content3) {
        Dialog dialog = new Dialog(this);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setContentView(R.layout.dialog_popup);
        dialog.setCancelable(true);


        if (dialog.getWindow() != null) {
            dialog.getWindow().setBackgroundDrawableResource(android.R.color.transparent);
            WindowManager windowManager = getWindowManager();
            Display display = windowManager.getDefaultDisplay();
            Point size = new Point();
            display.getSize(size);
            int width = size.x;
            int height = size.y;
            dialog.getWindow().setLayout((int) (width * 0.9), (int) (height * 0.8));
        }


        TextView txtPopupTitle = dialog.findViewById(R.id.txtPopupTitle);
        ImageView imgPopup = dialog.findViewById(R.id.imgPopup);
        TextView txtPopupContent = dialog.findViewById(R.id.txtSymptomContent);
        ImageButton btnClosePopup = dialog.findViewById(R.id.btnClosePopup);


        txtPopupTitle.setText(content);
        txtPopupContent.setText(content3); // Thêm nội dung chi tiết vào TextView


        int imageResId = getResources().getIdentifier(content2, "drawable", getPackageName());
        if (imageResId != 0) {
            imgPopup.setImageResource(imageResId);
        }


        btnClosePopup.setOnClickListener(v -> dialog.dismiss());


        dialog.show();
    }





    private void layDuLieuThoiTiet(String city) {
        String url = "https://api.openweathermap.org/data/2.5/weather?q=" + city + "&units=metric&appid=" + API_KEY;

        RequestQueue queue = Volley.newRequestQueue(this);

        JsonObjectRequest request = new JsonObjectRequest(Request.Method.GET, url, null,
                new Response.Listener<JSONObject>() {
                    @SuppressLint("SetTextI18n")
                    @Override
                    public void onResponse(JSONObject response) {
                        try {
                            JSONObject main = response.getJSONObject("main");
                            double nhietDo = main.getDouble("temp");
                            int doAm = main.getInt("humidity");

                            double khaNangMua = response.getJSONObject("clouds").getInt("all");

                            textViewNhietDo.setText("Nhiệt độ: " +nhietDo + "°C");
                            textViewKhaNangMua.setText("Mưa: " + khaNangMua + "%");
                            textViewDoAm.setText("Độ ẩm: " + doAm + "%");

                        } catch (JSONException e) {
                            Toast.makeText(HomeActivity.this, "Lỗi khi xử lý dữ liệu!", Toast.LENGTH_SHORT).show();
                        }
                    }
                }, new Response.ErrorListener() {
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(HomeActivity.this, "Không thể lấy dữ liệu!", Toast.LENGTH_SHORT).show();
            }
        });

        queue.add(request);
    }

}
