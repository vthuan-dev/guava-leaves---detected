package com.example.duannckh;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import android.os.Handler;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import org.json.JSONObject;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.json.JSONException;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_GALLERY = 1;
    private static final int REQUEST_CAMERA = 2;
    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_PICK_IMAGE = 2;
    private static final String TAG = "MainActivity";
    private static final int PERMISSION_REQUEST_CODE = 100;

    private ImageView imagePreview;
    private ProgressBar progressBar;
    private File imageFile;
    private Uri photoURI;
    private String currentPhotoPath;

    // Khai báo các Button với ID chính xác từ layout
    private Button btn_choose_image;
    private Button btn_capture_image;
    private Button btn_view_result;

    private ApiService apiService;
    private String prediction, treatment, medicine, medicineImage;
    private PyTorchClassifier classifier;
    private int userId;
    private String selectedImagePath;
    private Uri selectedImageUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Ánh xạ view với ID chính xác
        imagePreview = findViewById(R.id.imagePreview);
        progressBar = findViewById(R.id.progressBar);
        btn_choose_image = findViewById(R.id.btn_choose_image);
        btn_capture_image = findViewById(R.id.btn_capture_image);
        btn_view_result = findViewById(R.id.btn_view_result);
        
        // Khởi tạo classifier
        classifier = new PyTorchClassifier(this);

        // Khởi tạo ApiService
        createApiService();
        
        // Thiết lập ban đầu
        if (btn_view_result != null) {
            btn_view_result.setEnabled(false);
        }

        // Thiết lập sự kiện click
        setupClickListeners();
        
        // Kiểm tra quyền
        checkAndRequestPermissions();

        // Tìm và ẩn hai button ở cuối
        Button btnUseLoadedImage = findViewById(R.id.btn_use_loaded_image); // Cần thay ID chính xác
        Button btnUseGalleryImage = findViewById(R.id.btn_use_gallery_image); // Cần thay ID chính xác
        
        if (btnUseLoadedImage != null) {
            btnUseLoadedImage.setVisibility(View.GONE);
        }
        
        if (btnUseGalleryImage != null) {
            btnUseGalleryImage.setVisibility(View.GONE);
        }
    }

    private void createApiService() {
        try {
            OkHttpClient client = new OkHttpClient.Builder()
                .connectTimeout(60, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .writeTimeout(60, TimeUnit.SECONDS)
                .build();

            Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://nhandien-oi-d84e07b19a44.herokuapp.com/")
                .client(client)
                .addConverterFactory(GsonConverterFactory.create())
                .build();

            apiService = retrofit.create(ApiService.class);
            Log.d(TAG, "ApiService đã được khởi tạo thành công");
        } catch (Exception e) {
            Log.e(TAG, "Lỗi khi khởi tạo ApiService: " + e.getMessage(), e);
        }
    }

    private void setupClickListeners() {
        // Nút chọn ảnh từ thư viện
        if (btn_choose_image != null) {
            btn_choose_image.setOnClickListener(v -> {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, REQUEST_GALLERY);
            });
        }

        // Nút chụp ảnh từ camera
        if (btn_capture_image != null) {
            btn_capture_image.setOnClickListener(v -> {
                dispatchTakePictureIntent();
            });
        }
        
        // Nút xem kết quả
        if (btn_view_result != null) {
            btn_view_result.setOnClickListener(v -> {
                Intent intent = new Intent(MainActivity.this, ResultActivity.class);
                startActivity(intent);
            });
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        
        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_GALLERY && data != null) {
                Uri selectedImageUri = data.getData();
                Log.d(TAG, "onActivityResult: requestCode=" + requestCode + ", resultCode=" + resultCode + 
                      ", có data=" + (data != null) + ", có URI=" + (selectedImageUri != null));
                
                if (selectedImageUri != null) {
                    Log.d(TAG, "Đã chọn ảnh từ thư viện với URI: " + selectedImageUri.toString());
                    
                    try {
                        // Hiển thị ảnh
                        imagePreview.setImageURI(selectedImageUri);
                        Log.d(TAG, "Đã hiển thị ảnh lên ImageView");
                        
                        // Lưu ảnh vào file tạm
                        imageFile = createTempFileFromUri(selectedImageUri);
                        
                        // Hiển thị progress bar
                        if (progressBar != null) {
                            progressBar.setVisibility(View.VISIBLE);
                        }
                        
                        // Gửi ảnh đến API
                        if (apiService != null) {
                            sendImage(imageFile);
                        } else {
                            Log.e(TAG, "API Service chưa được khởi tạo");
                            Toast.makeText(this, "Lỗi kết nối API", Toast.LENGTH_SHORT).show();
                            progressBar.setVisibility(View.GONE);
                            
                            // Khởi tạo lại ApiService và thử lại
                            createApiService();
                            if (apiService != null) {
                                sendImage(imageFile);
                            } else {
                                Toast.makeText(this, "Không thể kết nối đến server", Toast.LENGTH_LONG).show();
                            }
                        }
                    } catch (Exception e) {
                        Log.e(TAG, "Lỗi xử lý ảnh: " + e.getMessage(), e);
                        Toast.makeText(this, "Lỗi xử lý ảnh: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                        if (progressBar != null) {
                            progressBar.setVisibility(View.GONE);
                        }
                    }
                }
            } else if (requestCode == REQUEST_CAMERA) {
                try {
                    // Hiển thị ảnh từ camera
                    Bitmap bitmap = BitmapFactory.decodeFile(currentPhotoPath);
                    imagePreview.setImageBitmap(bitmap);
                    
                    // Lưu đường dẫn ảnh
                    imageFile = new File(currentPhotoPath);
                    
                    // Hiển thị progress bar
                    if (progressBar != null) {
                        progressBar.setVisibility(View.VISIBLE);
                    }
                    
                    // Gửi ảnh đến API
                    if (apiService != null) {
                        sendImage(imageFile);
                    } else {
                        Log.e(TAG, "API Service chưa được khởi tạo");
                        Toast.makeText(this, "Lỗi kết nối API", Toast.LENGTH_SHORT).show();
                        progressBar.setVisibility(View.GONE);
                        
                        // Khởi tạo lại ApiService và thử lại
                        createApiService();
                        if (apiService != null) {
                            sendImage(imageFile);
                        } else {
                            Toast.makeText(this, "Không thể kết nối đến server", Toast.LENGTH_LONG).show();
                        }
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Lỗi xử lý ảnh từ camera: " + e.getMessage(), e);
                    Toast.makeText(this, "Lỗi xử lý ảnh: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                    if (progressBar != null) {
                        progressBar.setVisibility(View.GONE);
                    }
                }
            }
        }
    }

    private File createTempFileFromUri(Uri uri) throws IOException {
        InputStream inputStream = getContentResolver().openInputStream(uri);
        File tempFile = File.createTempFile("image", ".jpg", getCacheDir());
        FileOutputStream fos = new FileOutputStream(tempFile);
        
        byte[] buffer = new byte[1024];
        int bytesRead;
        while ((bytesRead = inputStream.read(buffer)) != -1) {
            fos.write(buffer, 0, bytesRead);
        }
        
        fos.close();
        inputStream.close();
        return tempFile;
    }

    private File resizeImage(File originalFile) {
        try {
            // Đọc ảnh từ file
            Bitmap originalBitmap = BitmapFactory.decodeFile(originalFile.getAbsolutePath());
            if (originalBitmap == null) {
                Log.e(TAG, "Không thể đọc ảnh từ file");
                return originalFile;
            }
            
            // Tính toán kích thước mới
            int maxSize = 1024; // Kích thước tối đa (pixel)
            int width = originalBitmap.getWidth();
            int height = originalBitmap.getHeight();
            float ratio = (float) width / height;
            
            if (width > height && width > maxSize) {
                width = maxSize;
                height = (int) (width / ratio);
            } else if (height > maxSize) {
                height = maxSize;
                width = (int) (height * ratio);
            }
            
            // Resize ảnh
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(originalBitmap, width, height, true);
            
            // Lưu ảnh đã resize
            File resizedFile = new File(getCacheDir(), "resized_" + originalFile.getName());
            FileOutputStream fos = new FileOutputStream(resizedFile);
            resizedBitmap.compress(Bitmap.CompressFormat.JPEG, 80, fos);
            fos.close();
            
            return resizedFile;
        } catch (Exception e) {
            Log.e(TAG, "Lỗi resize ảnh: " + e.getMessage(), e);
            return originalFile;
        }
    }

    private void sendImage(File file) {
        try {
            // Resize ảnh trước khi gửi để giảm kích thước
            File resizedFile = resizeImage(file);
            if (resizedFile == null) {
                showError("Không thể resize ảnh");
                progressBar.setVisibility(View.GONE);
                return;
            }
            
            Log.d(TAG, "Sending file: " + file.getAbsolutePath() + ", size: " + file.length() + " bytes");
            Log.d(TAG, "Resized file: " + resizedFile.getAbsolutePath() + ", size: " + resizedFile.length() + " bytes");
            
            // Tạo request
            RequestBody requestFile = RequestBody.create(MediaType.parse("image/jpeg"), resizedFile);
            MultipartBody.Part body = MultipartBody.Part.createFormData("file", resizedFile.getName(), requestFile);
            
            // Gửi request - Sử dụng phương thức API đúng
            Call<ResponseBody> call = apiService.predict(body);
            
            call.enqueue(new Callback<ResponseBody>() {
                @Override
                public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                    progressBar.setVisibility(View.GONE);
                    try {
                        if (response.isSuccessful() && response.body() != null) {
                            String jsonData = response.body().string();
                            handleApiResponse(jsonData);
                        } else {
                            Toast.makeText(MainActivity.this, "Lỗi từ máy chủ: " + response.code(), Toast.LENGTH_SHORT).show();
                        }
                    } catch (IOException e) {
                        Log.e(TAG, "Lỗi đọc phản hồi: " + e.getMessage());
                        Toast.makeText(MainActivity.this, "Lỗi đọc phản hồi", Toast.LENGTH_SHORT).show();
                    }
                }
                
                @Override
                public void onFailure(Call<ResponseBody> call, Throwable t) {
                    progressBar.setVisibility(View.GONE);
                    Log.e(TAG, "Network error: " + t.getMessage(), t);
                    showError("Lỗi kết nối: " + t.getMessage());
                }
            });
        } catch (Exception e) {
            progressBar.setVisibility(View.GONE);
            Log.e(TAG, "Error sending image: " + e.getMessage(), e);
            showError("Lỗi gửi ảnh: " + e.getMessage());
        }
    }

    private void showError(String message) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
    }

    private void checkAndRequestPermissions() {
        String[] permissions = {
            android.Manifest.permission.CAMERA,
            android.Manifest.permission.READ_EXTERNAL_STORAGE,
            android.Manifest.permission.WRITE_EXTERNAL_STORAGE
        };

        ArrayList<String> permissionsNeeded = new ArrayList<>();
        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                permissionsNeeded.add(permission);
            }
        }

        if (!permissionsNeeded.isEmpty()) {
            ActivityCompat.requestPermissions(this, 
                permissionsNeeded.toArray(new String[0]), 
                PERMISSION_REQUEST_CODE);
        }
    }

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            File photoFile = null;
            try {
                photoFile = createImageFile();
            } catch (IOException ex) {
                Log.e(TAG, "Lỗi tạo file ảnh: " + ex.getMessage());
                Toast.makeText(this, "Lỗi tạo file ảnh", Toast.LENGTH_SHORT).show();
            }
            
            if (photoFile != null) {
                photoURI = FileProvider.getUriForFile(this,
                        "com.example.duannckh.fileprovider",
                        photoFile);
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(takePictureIntent, REQUEST_CAMERA);
            }
        }
    }

    private File createImageFile() throws IOException {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );
        currentPhotoPath = image.getAbsolutePath();
        return image;
    }

    // Phương thức xử lý kết quả API trong MainActivity
    private void handleApiResponse(String responseBody) {
        try {
            Log.d(TAG, "API Response: " + responseBody);
            
            // Parse JSON response
            JSONObject jsonResponse = new JSONObject(responseBody);
            
            // Lấy kết quả từ mô hình ensemble
            JSONObject ensembleResult = jsonResponse.getJSONObject("ensemble");
            String predictedClass = ensembleResult.getString("predicted_class");
            double confidence = ensembleResult.getDouble("confidence");
            
            // Định dạng độ tin cậy
            String formattedConfidence = String.format("%.2f%%", confidence * 100);
            
            // Lưu kết quả vào ResultData singleton
            ResultData.getInstance().setPrediction(predictedClass);
            ResultData.getInstance().setConfidence(formattedConfidence);
            
            // Thêm thông tin điều trị dựa trên loại bệnh
            String treatment = getTreatmentForDisease(predictedClass);
            ResultData.getInstance().setTreatment(treatment);
            
            // Thêm thông tin thuốc dựa trên loại bệnh
            String medicine = getMedicineForDisease(predictedClass);
            ResultData.getInstance().setMedicine(medicine);
            
            // Lưu đường dẫn hình ảnh
            if (currentPhotoPath != null) {
                ResultData.getInstance().setImagePath(currentPhotoPath);
            } else if (imageFile != null) {
                ResultData.getInstance().setImagePath(imageFile.getAbsolutePath());
            }
            
            Log.d(TAG, "Đã lưu kết quả phân loại: " + predictedClass + " với độ tin cậy: " + formattedConfidence);
            
            // Chuyển đến màn hình kết quả
            Intent intent = new Intent(MainActivity.this, ResultActivity.class);
            startActivity(intent);
            
        } catch (JSONException e) {
            Log.e(TAG, "Lỗi xử lý JSON: " + e.getMessage());
            Toast.makeText(this, "Lỗi xử lý kết quả từ máy chủ", Toast.LENGTH_SHORT).show();
        }
    }

    // Phương thức để lấy cách điều trị dựa trên loại bệnh
    private String getTreatmentForDisease(String disease) {
        if (disease == null) return "";
        
        switch (disease) {
            case "Bệnh loét":
                return "1. Loại bỏ lá bị bệnh\n2. Phun thuốc fungicide có chứa đồng\n3. Cải thiện thoát nước và thông gió";
            case "Bệnh đốm":
                return "1. Loại bỏ lá bị bệnh\n2. Phun thuốc fungicide định kỳ\n3. Tránh tưới nước trên lá";
            case "Bệnh rỉ sắt":
                return "1. Loại bỏ lá bị bệnh\n2. Phun thuốc fungicide chứa sulfur\n3. Luân canh cây trồng";
            case "Bệnh thán thư":
                return "1. Loại bỏ lá, quả bị bệnh\n2. Phun thuốc copper fungicide\n3. Tăng cường thông gió";
            case "Khỏe mạnh":
                return "Cây đang khỏe mạnh, tiếp tục chăm sóc đúng cách";
            default:
                return "Không có thông tin điều trị cho bệnh này";
        }
    }

    // Phương thức để lấy thuốc điều trị dựa trên loại bệnh
    private String getMedicineForDisease(String disease) {
        if (disease == null) return "";
        
        switch (disease) {
            case "Bệnh loét":
                return "1. Copper oxychloride 50WP\n2. Bordeaux mixture\n3. Mancozeb 75WP";
            case "Bệnh đốm":
                return "1. Chlorothalonil\n2. Propineb 70WP\n3. Azoxystrobin 25SC";
            case "Bệnh rỉ sắt":
                return "1. Sulfur WP\n2. Propiconazole 25EC\n3. Tebuconazole 25WP";
            case "Bệnh thán thư":
                return "1. Copper fungicides\n2. Carbendazim 50WP\n3. Difenoconazole 25EC";
            case "Khỏe mạnh":
                return "Không cần thuốc điều trị";
            default:
                return "Không có thông tin thuốc cho bệnh này";
        }
    }
}













