# Biweekly Meeting 3 - Heart Risk Prediction Project

**Date:** January 9, 2026  
**Duration:** 2 hours  
**Attendees:** Research Team, Prof. Dr. Beate Rhein, Mr. Håkan Lane (Nightingale Heart)

## Sprint 5-6 Progress Summary

### Agenda
- Week 5-6 XAI implementation results
- SHAP analysis findings and clinical insights
- Root cause analysis of optimization paradox
- Clinical deployment assessment update

### Progress Updates

#### Completed Tasks ✅
- ✅ **SHAP Implementation**: Complete TreeExplainer setup with 500 test samples
- ✅ **Global Feature Importance**: BMI (0.0208) and exercise (0.0189) as top clinical predictors
- ✅ **Clinical Interpretation**: Psychological features identified as weak predictors
- ✅ **Visualization Suite**: Feature importance, summary plots, individual patient analysis
- ✅ **Root Cause Validation**: Dataset limitation confirmed via XAI analysis
- ✅ **Clinical Assessment**: XAI confirms deployment safety concerns
- ✅ **Documentation Update**: Comprehensive results integration in final report

#### Key XAI Findings
- **Optimization Paradox Explained**: Weak psychological predictors cannot be optimized for clinical performance
- **Feature Quality Gap**: Missing traditional cardiac risk factors (ECG, chest pain, family history)
- **Clinical Reality**: Models attempt prediction from lifestyle surveys, not medical assessments
- **Safety Validation**: SHAP confirms 2.1% sensitivity insufficient for deployment

#### Blockers/Issues
- **Resolved**: Fixed SHAP variable reference errors in notebook execution
- **None Currently**: All Week 5-6 objectives successfully completed

### Decisions Made
- **XAI Success**: SHAP provides excellent clinical interpretability confirming research hypotheses
- **Clinical Recommendation**: Use only as lifestyle screening tool, not diagnostic aid
- **Research Value**: Honest assessment of ML limitations provides significant academic contribution
- **Final Phase**: Ready for clinical decision support template development

### Major Achievements

#### Technical Implementation
- **SHAP Framework**: 15-cell comprehensive explainability notebook
- **Feature Analysis**: Top 10 clinical features with medical interpretation
- **Individual Cases**: 3 representative patient explanation scenarios identified
- **Performance Validation**: Baseline RF sensitivity 2.1% confirmed via XAI

#### Clinical Insights
- **Physical Health Valid**: BMI and exercise show excellent clinical correlation
- **Psychological Limitation**: Happiness/mood features drive decisions but lack predictive power
- **Traditional Missing**: ECG, blood pressure, cholesterol data absent from dataset
- **Safety Assessment**: All models fail minimum deployment criteria (≥80% sensitivity)

### Action Items
| Task | Assignee | Due Date | Status |
|------|----------|----------|--------|
| Final report Week 5-6 integration | Research Team | January 10 | ✅ Complete |
| Clinical decision support templates | Research Team | January 15 | In Progress |
| README project structure update | Research Team | January 10 | In Progress |
| Academic paper preparation | Research Team | January 20 | Planned |

### Next Sprint Goals
- Finalize clinical decision support framework
- Complete comprehensive documentation
- Prepare academic publication materials
- Industry partner presentation preparation

### Notes
- **Week 5-6 Success**: XAI implementation exceeded expectations in clinical interpretability
- **Research Impact**: First comprehensive study combining optimization failure with XAI validation
- **Clinical Value**: Clear explanation of why psychological factors insufficient for cardiac prediction
- **Academic Contribution**: Honest assessment methodology for healthcare ML deployment challenges