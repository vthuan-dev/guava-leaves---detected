package com.example.duannckh;

import okhttp3.MultipartBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;

public interface ApiService {
    @Multipart
    @POST("predict")
    Call<ResponseBody> predict(@Part MultipartBody.Part file);

    @Multipart
    @POST("classify")
    Call<ResponseBody> classify(@Part MultipartBody.Part file);

    @Multipart
    @POST("api/classify")
    Call<ResponseBody> apiClassify(@Part MultipartBody.Part file);
}