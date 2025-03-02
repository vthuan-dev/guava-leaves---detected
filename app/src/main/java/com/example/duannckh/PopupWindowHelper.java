package com.example.duannckh;


import android.app.Activity;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.Button;
import android.widget.PopupWindow;
import android.view.ViewGroup;
import android.view.WindowManager;


public class PopupWindowHelper {
    private PopupWindow popupWindow;


    public void showPopup(Activity activity, View anchorView) {
        // Inflate layout của popup
        View popupView = LayoutInflater.from(activity).inflate(R.layout.dialog_popup, null);


        // Tạo PopupWindow
        popupWindow = new PopupWindow(popupView, ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, true);


        // Đặt vị trí popup
        popupWindow.showAtLocation(anchorView, Gravity.CENTER, 0, 0);


        // Làm mờ màn hình phía sau Popup
        WindowManager.LayoutParams layoutParams = activity.getWindow().getAttributes();
        layoutParams.alpha = 0.5f;
        activity.getWindow().setAttributes(layoutParams);


        // Xử lý khi đóng Popup
        Button btnClose = popupView.findViewById(R.id.btnClosePopup);
        btnClose.setOnClickListener(v -> {
            popupWindow.dismiss();
        });


        // Khi Popup đóng, khôi phục độ sáng màn hình
        popupWindow.setOnDismissListener(() -> {
            layoutParams.alpha = 1f;
            activity.getWindow().setAttributes(layoutParams);
        });
    }
}



