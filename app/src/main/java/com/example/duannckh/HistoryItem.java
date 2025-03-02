
package com.example.duannckh;

public class HistoryItem {
    private String imageName;
    private String dateTime;
    private String diseaseName;
    private String medicineName;
    private String medicineImage;

    public HistoryItem(String imageName, String dateTime, String diseaseName, String medicineName, String medicineImage) {
        this.imageName = imageName;
        this.dateTime = dateTime;
        this.diseaseName = diseaseName;
        this.medicineName = medicineName;
        this.medicineImage = medicineImage;
    }

    public String getImageName() {
        return imageName;
    }

    public String getDateTime() {
        return dateTime;
    }

    public String getDiseaseName() {
        return diseaseName;
    }

    public String getMedicineName() {
        return medicineName;
    }

    public String getMedicineImage() {
        return medicineImage;
    }
}
