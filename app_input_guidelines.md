# Heart Risk Prediction App - Input Guidelines

## Risk Level Guidelines Based on Current App Settings

This document provides specific input ranges that generate different risk levels in the Heart Disease Risk Prediction App.

---

## 🟢 **LOW RISK (< 25% probability)**

**To get LOW risk output, use these input ranges:**

### Personal Information
- **Age:** 20-35 years
- **Height:** Any (170-180 cm recommended)
- **Weight:** Proportional to maintain BMI 18-23
- **BMI Target:** 18-23 (normal weight range)

### Lifestyle Factors (0-10 scale)
- **Physical Activity Level:** 7-10 (high activity)
- **Smoking Intensity:** 0-1 (non-smoker to very light)
- **Alcohol Consumption:** 0-2 (minimal to light)
- **Fruit & Vegetable Intake:** 7-10 (high intake)

### Wellbeing Assessment (0-10 scale)
- **Overall Life Satisfaction:** 8-10 (very satisfied)
- **Sense of Control Over Life:** 8-10 (high control)
- **Social Engagement Level:** 7-10 (very social)
- **Sleep Quality:** 7-10 (excellent sleep)

### Example LOW RISK Profile:
```
Age: 25, Height: 175cm, Weight: 65kg (BMI: 21.2)
Exercise: 9, Smoking: 0, Alcohol: 1, Fruit Intake: 9
Happiness: 9, Life Control: 9, Social: 8, Sleep: 9
Expected Result: 🟢 Low Risk (~19% probability)
```

---

## 🟡 **MODERATE RISK (25-34.9% probability)**

**To get MODERATE risk output, use these input ranges:**

### Personal Information
- **Age:** 35-55 years
- **Height:** Any (165-180 cm)
- **Weight:** Proportional to maintain BMI 24-29
- **BMI Target:** 24-29 (overweight range)

### Lifestyle Factors (0-10 scale)
- **Physical Activity Level:** 3-6 (moderate activity)
- **Smoking Intensity:** 2-5 (light to moderate smoking)
- **Alcohol Consumption:** 3-5 (moderate drinking)
- **Fruit & Vegetable Intake:** 3-6 (moderate intake)

### Wellbeing Assessment (0-10 scale)
- **Overall Life Satisfaction:** 4-7 (average satisfaction)
- **Sense of Control Over Life:** 4-7 (moderate control)
- **Social Engagement Level:** 3-6 (moderate social activity)
- **Sleep Quality:** 4-6 (fair to good sleep)

### Example MODERATE RISK Profile:
```
Age: 45, Height: 170cm, Weight: 78kg (BMI: 27.0)
Exercise: 5, Smoking: 3, Alcohol: 4, Fruit Intake: 5
Happiness: 6, Life Control: 6, Social: 5, Sleep: 6
Expected Result: 🟡 Moderate Risk (~26-32% probability)
```

---

## 🔴 **HIGH RISK (≥ 35% probability)**

**HIGH risk occurs with these input ranges:**

### Personal Information
- **Age:** 55+ years (especially 60+)
- **Height:** Any (160-175 cm)
- **Weight:** High relative to height
- **BMI Target:** 30+ (obese range)

### Lifestyle Factors (0-10 scale)
- **Physical Activity Level:** 0-2 (sedentary lifestyle)
- **Smoking Intensity:** 6-10 (heavy smoking)
- **Alcohol Consumption:** 6-10 (heavy drinking)
- **Fruit & Vegetable Intake:** 0-2 (poor diet)

### Wellbeing Assessment (0-10 scale)
- **Overall Life Satisfaction:** 1-4 (low satisfaction)
- **Sense of Control Over Life:** 1-4 (low control)
- **Social Engagement Level:** 1-3 (socially isolated)
- **Sleep Quality:** 1-4 (poor sleep)

### Example HIGH RISK Profile:
```
Age: 65, Height: 165cm, Weight: 95kg (BMI: 34.9)
Exercise: 1, Smoking: 8, Alcohol: 7, Fruit Intake: 1
Happiness: 3, Life Control: 3, Social: 2, Sleep: 3
Expected Result: 🔴 High Risk (~36-52% probability)
```

---

## 💡 **Testing Tips for Demonstrations**

### Quick Test Combinations:

**For LOW Risk Demo:**
- Young person: Age 25-30, Exercise 8-9, Smoking 0, Happiness 8-9
- Healthy lifestyle with normal BMI

**For MODERATE Risk Demo:**
- Middle-aged: Age 40-50, Exercise 4-5, Smoking 2-3, moderate lifestyle
- Slightly overweight BMI

**For HIGH Risk Demo:**
- Older person: Age 60+, Exercise 1-2, Smoking 7-8, poor lifestyle habits
- Obese BMI with multiple risk factors

---

## ⚖️ **Current Risk Thresholds**

The app uses these probability thresholds:
- **Low Risk:** < 25% probability
- **Moderate Risk:** 25% - 34.9% probability  
- **High Risk:** ≥ 35% probability

---

## 📝 **Notes for Meetings/Demonstrations**

1. **The app now properly shows all three risk levels** based on input variations
2. **Use these EXACT values for reliable demonstrations:**

### **✅ LOW RISK Demo Values:**
```
Age: 25, Height: 175, Weight: 65
Exercise: 9, Smoking: 0, Alcohol: 1, Fruit: 9
Happiness: 9, Life Control: 9, Social: 8, Sleep: 9
Expected: 🟢 Low Risk (~18-19%)
```

### **✅ MODERATE RISK Demo Values:**
```
Age: 45, Height: 170, Weight: 78
Exercise: 5, Smoking: 3, Alcohol: 4, Fruit: 5  
Happiness: 6, Life Control: 6, Social: 5, Sleep: 6
Expected: 🟡 Moderate Risk (~27-28%)
```

### **✅ HIGH RISK Demo Values:**
```
Age: 65, Height: 165, Weight: 95
Exercise: 1, Smoking: 8, Alcohol: 7, Fruit: 1
Happiness: 3, Life Control: 3, Social: 2, Sleep: 3  
Expected: 🔴 High Risk (~36-37%)
```

3. **Important**: If you're still getting High Risk for all inputs, **restart the Docker container** as it may be using cached old code
4. **Test locally first**: Run `python app/app_gradio.py` to verify the updated risk classification works
5. **Docker rebuild**: If needed, run `docker build -t heart-risk-final -f docker/Dockerfile .` and restart

**App URL:** http://localhost:7860

*Last updated: January 19, 2026*