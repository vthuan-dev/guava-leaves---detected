package com.example.duannckh;

import android.annotation.SuppressLint;
import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;


public class DatabaseHelper extends SQLiteOpenHelper {

    // Khai báo tên cơ sở dữ liệu và phiên bản
    private static final String DATABASE_NAME = "UserDatabase.db";
    private static final int DATABASE_VERSION = 2;

    // Tên bảng User và các cột của nó
    private static final String TABLE_USER = "User";
    private static final String COLUMN_ID = "id";
    private static final String COLUMN_FULL_NAME = "full_name";
    private static final String COLUMN_USERNAME = "username";
    private static final String COLUMN_EMAIL = "email";
    private static final String COLUMN_PASSWORD = "password";

    // Tên bảng History và các cột của nó
    private static final String TABLE_HISTORY = "History";
    private static final String COLUMN_HISTORY_ID = "history_id";
    private static final String COLUMN_DISEASE = "disease";
    public static final String COLUMN_TREATMENT = "treatment";
    private static final String COLUMN_IMAGE_NAME = "image_name";
    private static final String COLUMN_DATE_TIME = "date_time";
    public static final String COLUMN_TREATMENT_DRUG = "treatment_drug";
    public static final String COLUMN_IMAGE_NAME_MEDICINE = "image_medicine";

    // Tên bảng Diseases và các cột của nó
    public static final String TABLE_DISEASES = "Diseases";
    private static final String COLUMN_DISEASE_ID = "disease_id";
    public static final String COLUMN_DISEASE_NAME = "disease_name";
    public static final String COLUMN_DISEASE_DESCRIPTION = "disease_description";

    public DatabaseHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        // Tạo bảng User
        String createTableUser = "CREATE TABLE " + TABLE_USER + " ("
                + COLUMN_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, "
                + COLUMN_FULL_NAME + " TEXT, "
                + COLUMN_USERNAME + " TEXT, "
                + COLUMN_EMAIL + " TEXT, "
                + COLUMN_PASSWORD + " TEXT)";
        db.execSQL(createTableUser);

        // Tạo bảng History
        String createTableHistory = "CREATE TABLE " + TABLE_HISTORY + " ("
                + COLUMN_ID + " INTEGER, "
                + COLUMN_HISTORY_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, "
                + COLUMN_DISEASE + " TEXT, "
                + COLUMN_TREATMENT + " TEXT, "
                + COLUMN_IMAGE_NAME + " TEXT, "
                + COLUMN_DATE_TIME + " TEXT, "
                + COLUMN_TREATMENT_DRUG + " TEXT,"
                + COLUMN_IMAGE_NAME_MEDICINE + " TEXT)";
        db.execSQL(createTableHistory);

        // Tạo bảng Diseases
        String createTableDiseases = "CREATE TABLE " + TABLE_DISEASES + " ("
                + COLUMN_DISEASE_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, "
                + COLUMN_DISEASE_NAME + " TEXT, "
                + COLUMN_DISEASE_DESCRIPTION + " TEXT)";
        db.execSQL(createTableDiseases);

        // Thêm dữ liệu mẫu vào bảng Diseases
        db.execSQL("INSERT INTO " + TABLE_DISEASES + " (" + COLUMN_DISEASE_NAME + ", " + COLUMN_DISEASE_DESCRIPTION + ") VALUES ('Anthracnose', 'Bệnh thán thư là một bệnh phổ biến trên cây ổi...')");
        db.execSQL("INSERT INTO " + TABLE_DISEASES + " (" + COLUMN_DISEASE_NAME + ", " + COLUMN_DISEASE_DESCRIPTION + ") VALUES ('Powdery Mildew', 'Bệnh phấn trắng là một bệnh gây hại trên lá và quả...')");
        // Thêm các bệnh khác nếu cần
    }
    // Phương thức getRandomDisease
    @SuppressLint("Range")
    public String[] getRandomDisease() {
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor cursor = db.rawQuery("SELECT * FROM " + TABLE_DISEASES + " ORDER BY RANDOM() LIMIT 1", null);
        String[] disease = new String[2];
        if (cursor.moveToFirst()) {
            disease[0] = cursor.getString(cursor.getColumnIndex(COLUMN_DISEASE_NAME));
            disease[1] = cursor.getString(cursor.getColumnIndex(COLUMN_DISEASE_DESCRIPTION));
        }
        cursor.close();
        return disease;
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        // Xóa bảng cũ nếu có và tạo bảng mới
        db.execSQL("DROP TABLE IF EXISTS " + TABLE_USER);
        db.execSQL("DROP TABLE IF EXISTS " + TABLE_HISTORY);
        db.execSQL("DROP TABLE IF EXISTS " + TABLE_DISEASES);
        onCreate(db);
    }

    public boolean addUser(String fullName, String username, String email, String password) {
        SQLiteDatabase db = this.getWritableDatabase();
        ContentValues values = new ContentValues();
        values.put(COLUMN_FULL_NAME, fullName);
        values.put(COLUMN_USERNAME, username);
        values.put(COLUMN_EMAIL, email);
        values.put(COLUMN_PASSWORD, password);

        long result = db.insert(TABLE_USER, null, values);
        db.close();

        // Nếu thêm thành công, result sẽ khác -1
        return result != -1;
    }

    public boolean checkUsernameExists(String username) {
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor cursor = db.query(TABLE_USER, new String[]{COLUMN_ID},
                COLUMN_USERNAME + "=?", new String[]{username},
                null, null, null);
        boolean exists = cursor.getCount() > 0;
        cursor.close();
        db.close();
        return exists;
    }


    public boolean checkUser(String username, String password) {
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor cursor = db.query(TABLE_USER,
                new String[]{COLUMN_ID},
                COLUMN_USERNAME + "=? AND " + COLUMN_PASSWORD + "=?",
                new String[]{username, password},
                null, null, null);

        boolean exists = cursor.getCount() > 0;
        cursor.close();
        db.close();
        return exists;
    }


    // Hàm để thêm bản ghi vào bảng History
    public boolean addHistory(String columnid, String disease, String treatment, String image_name, String dateTime,String treatment_drug, String image_medicine) {
        SQLiteDatabase db = this.getWritableDatabase();
        ContentValues values = new ContentValues();
        values.put(COLUMN_ID, columnid);
        values.put(COLUMN_DISEASE, disease);
        values.put(COLUMN_TREATMENT, treatment);
        values.put(COLUMN_IMAGE_NAME, image_name);
        values.put(COLUMN_DATE_TIME, dateTime);
        values.put(COLUMN_TREATMENT_DRUG, treatment_drug);
        values.put(COLUMN_IMAGE_NAME_MEDICINE, image_medicine);


        long result = db.insert(TABLE_HISTORY, null, values);
        db.close();
        // Kiểm tra kết quả
        if (result == -1) {
            Log.e("DatabaseHelper", "Failed to insert record into History table");
            return false;
        } else {
            return true;
        }
    }




    public Cursor getAllHistory() {
        SQLiteDatabase db = this.getReadableDatabase();
        // Sắp xếp theo `history_id` thay vì `id`
        return db.rawQuery("SELECT * FROM " + TABLE_HISTORY + " ORDER BY " + COLUMN_HISTORY_ID + " DESC", null);
    }
    // Phương thức để lấy thông tin disease và treatment theo image_name
    public Cursor getHistoryByImageName(String imageName) {
        SQLiteDatabase db = this.getReadableDatabase();
        String query = "SELECT * FROM " + TABLE_HISTORY + " WHERE " + COLUMN_IMAGE_NAME + " = ?";
        return db.rawQuery(query, new String[]{imageName});
    }
    @SuppressLint("Range")
    public int getUserId(String username) {
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor cursor = db.query(TABLE_USER,
                new String[]{COLUMN_ID},
                COLUMN_USERNAME + "=?",
                new String[]{username},
                null, null, null);




        int userId = -1;
        if (cursor.moveToFirst()) {
            userId = cursor.getInt(cursor.getColumnIndex(COLUMN_ID));
        }
        cursor.close();
        db.close();
        return userId;
    }
    public void deleteHistoryByImageName(String imageName) {
        SQLiteDatabase db = this.getWritableDatabase();
        db.delete(TABLE_HISTORY, COLUMN_IMAGE_NAME + " = ?", new String[]{imageName});
        db.close();
    }




    public void clearUserHistory(int userId) {
        SQLiteDatabase db = this.getWritableDatabase();
        db.delete("history", COLUMN_ID + " = ?", new String[]{String.valueOf(userId)});
        db.close();
    }

}
