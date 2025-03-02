package com.example.duannckh;

public class ResultData {
    private static ResultData instance;
    
    private String prediction;
    private String confidence;
    private String treatment;
    private String medicine;
    private String imagePath;
    
    private ResultData() {
        // Private constructor to enforce singleton pattern
    }
    
    public static synchronized ResultData getInstance() {
        if (instance == null) {
            instance = new ResultData();
        }
        return instance;
    }
    
    public String getPrediction() {
        return prediction;
    }
    
    public void setPrediction(String prediction) {
        this.prediction = prediction;
    }
    
    public String getConfidence() {
        return confidence;
    }
    
    public void setConfidence(String confidence) {
        this.confidence = confidence;
    }
    
    public String getTreatment() {
        return treatment;
    }
    
    public void setTreatment(String treatment) {
        this.treatment = treatment;
    }
    
    public String getMedicine() {
        return medicine;
    }
    
    public void setMedicine(String medicine) {
        this.medicine = medicine;
    }
    
    public String getImagePath() {
        return imagePath;
    }
    
    public void setImagePath(String imagePath) {
        this.imagePath = imagePath;
    }
} 