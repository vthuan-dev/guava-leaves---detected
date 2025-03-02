package com.example.duannckh;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.style.ForegroundColorSpan;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;

public class RegisterActivity extends AppCompatActivity {

    private EditText etFullName, etUsername, etEmail, etPassword, etConfirmPassword;
    //Khai báo các biến EditText để nhập liệu cho tên đầy đủ,
    // tên đăng nhập, email, mật khẩu, và xác nhận mật khẩu.
    private Button btnRegister;//Khai báo biến Button cho nút đăng ký.
    private TextView tvLogin;//Khai báo biến TextView cho liên kết đến màn hình đăng nhập.
    private DatabaseHelper db;//Khai báo đối tượng DatabaseHelper để tương tác với cơ sở dữ liệu.

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.register_activity);

        tvLogin = findViewById(R.id.tv_login);
        String text = "Đã có tài khoản? Đăng nhập";
        SpannableString spannable = new SpannableString(text);

        // Định dạng màu đen cho phần "Đã có tài khoản?"
        spannable.setSpan(new ForegroundColorSpan(Color.BLACK), 0, 17, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);

        // Định dạng màu xanh lá cho phần "Đăng nhập"
        spannable.setSpan(new ForegroundColorSpan(Color.parseColor("#009900")), 17, text.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);

        tvLogin.setText(spannable);

        // Đặt giao diện người dùng của Activity theo tệp XML register_activity.
        // Khởi tạo DatabaseHelper để quản lý cơ sở dữ liệu.
        db = new DatabaseHelper(this);

        // Khởi tạo các EditText và Button
        etFullName = findViewById(R.id.et_full_name);
        etUsername = findViewById(R.id.et_username);
        etEmail = findViewById(R.id.et_email);
        etPassword = findViewById(R.id.et_password);
        etConfirmPassword = findViewById(R.id.et_confirm_password);
        btnRegister = findViewById(R.id.btn_register);

        // Liên kết các thành phần trong giao diện người dùng với các biến đã khai báo.
        btnRegister.setOnClickListener(v -> {
            String fullName = etFullName.getText().toString().trim();
            String username = etUsername.getText().toString().trim();
            String email = etEmail.getText().toString().trim();
            String password = etPassword.getText().toString().trim();
            String confirmPassword = etConfirmPassword.getText().toString().trim();

            if (fullName.isEmpty() || username.isEmpty() || email.isEmpty() || password.isEmpty()) {
                Toast.makeText(RegisterActivity.this, "Vui lòng điền đầy đủ thông tin", Toast.LENGTH_SHORT).show();
            } else if (!password.equals(confirmPassword)) {
                Toast.makeText(RegisterActivity.this, "Mật khẩu xác nhận không khớp", Toast.LENGTH_SHORT).show();
            } else if (db.checkUsernameExists(username)) {
                Toast.makeText(RegisterActivity.this, "Tên đăng nhập đã tồn tại", Toast.LENGTH_SHORT).show();
            } else {
                boolean isInserted = db.addUser(fullName, username, email, password);
                if (isInserted) {
                    Toast.makeText(RegisterActivity.this, "Đăng ký thành công", Toast.LENGTH_SHORT).show();
                    Intent intent = new Intent(RegisterActivity.this, LoginActivity.class);
                    startActivity(intent);
                    finish();
                } else {
                    Toast.makeText(RegisterActivity.this, "Đăng ký thất bại", Toast.LENGTH_SHORT).show();
                }
            }
        });

        tvLogin.setOnClickListener(v -> {
            Intent intent = new Intent(RegisterActivity.this, LoginActivity.class);
            startActivity(intent);
            finish();
        });
    }
}
