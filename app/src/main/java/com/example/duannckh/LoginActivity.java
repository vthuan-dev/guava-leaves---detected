package com.example.duannckh;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.style.ForegroundColorSpan;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;

public class LoginActivity extends AppCompatActivity {

    private EditText etUsername, etPassword;
    private Button btnLogin;
    private TextView tvRegister;
    private DatabaseHelper db;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.login_activity);
        TextView tvRegister = findViewById(R.id.tv_register);
        String text = "Chưa có tài khoản? Đăng ký ngay";

        SpannableString spannable = new SpannableString(text);

// Định dạng màu đen cho phần "Chưa có tài khoản?"
        spannable.setSpan(new ForegroundColorSpan(Color.WHITE), 0, 18, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);

// Định dạng màu xanh dương cho phần "Đăng ký ngay"
        spannable.setSpan(new ForegroundColorSpan(Color.parseColor("#009900")), 19, text.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);

        tvRegister.setText(spannable);

        // Khởi tạo DatabaseHelper
        db = new DatabaseHelper(this);

        etUsername = findViewById(R.id.et_username);
        etPassword = findViewById(R.id.et_password);
        btnLogin = findViewById(R.id.btn_login);
        tvRegister = findViewById(R.id.tv_register);

        // Xử lý khi nhấn vào nút Đăng nhập
        btnLogin.setOnClickListener(v -> {
            try {
                String username = etUsername.getText().toString().trim();
                String password = etPassword.getText().toString().trim();

                if (db.checkUser(username, password)) {  // Kiểm tra với cơ sở dữ liệu
                    int userId = db.getUserId(username);  // Lấy COLUMN_ID của người dùng
                    Toast.makeText(LoginActivity.this, "Đăng nhập thành công. ID: " + userId, Toast.LENGTH_SHORT).show();

                    Intent intent = new Intent(LoginActivity.this, HomeActivity.class);
                    intent.putExtra("userId", userId);  // Truyền COLUMN_ID qua HomeActivity nếu cần
                    startActivity(intent);
                    finish();  // Kết thúc LoginActivity để không quay lại khi nhấn nút Back
                } else {
                    Toast.makeText(LoginActivity.this, "Sai tên đăng nhập hoặc mật khẩu", Toast.LENGTH_SHORT).show();
                }
            } catch (Exception e) {
                Log.e("LoginActivity", "Error in login process", e);
                Toast.makeText(LoginActivity.this, "Đã xảy ra lỗi.", Toast.LENGTH_SHORT).show();
            }
        });



        // Điều hướng sang form đăng ký nếu người dùng chưa có tài khoản
        tvRegister.setOnClickListener(v -> {

            Intent intent = new Intent(LoginActivity.this, RegisterActivity.class);
            startActivity(intent);
        });
    }
}