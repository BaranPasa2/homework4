import pickle

rating_vars = {
    "2008": ["contractid",
                   "contract_name",
                   "breastcancer_screen",
                   "rectalcancer_screen",
                   "cv_cholscreen",
                   "diabetes_cholscreen",
                   "glaucoma_test",
                   "monitoring",
                   "flu_vaccine",
                   "pn_vaccine",
                   "primaryaccess",
                   "hospital_followup",
                   "depression_followup",
                   "nodelays",
                   "carequickly",
                   "overallrating_care",
                   "overallrating_plan",
                   "calltime",
                   "doctor_communicate",
                   "osteo_manage",
                   "diabetes_eye",
                   "diabetes_kidney",
                   "diabetes_bloodsugar",
                   "diabetes_chol",
                   "antidepressant", 
                   "bloodpressure",
                   "ra_manage",
                   "copd_test",
                   "betablocker",
                   "appeals_timely",
                   "appeals_review",
                   "new_contract"],
    "2009": ["contractid",
                   "org_type",
                   "contract_name",
                   "org_marketing",
                   "breastcancer_screen",
                   "rectalcancer_screen",
                   "cv_cholscreen",
                   "diabetes_cholscreen",
                   "glaucoma_test",
                   "monitoring",
                   "flu_vaccine",
                   "pn_vaccine",
                   "physical_health",
                   "mental_health",
                   "osteo_test",
                   "physical_monitor",
                   "primaryaccess",
                   "hospital_followup",
                   "depression_followup",
                   "nodelays",
                   "carequickly",
                   "overallrating_care",
                   "overallrating_plan",
                   "calltime",
                   "doctor_communicate",
                   "customer_service",
                   "osteo_manage",
                   "diabetes_eye",
                   "diabetes_kidney",
                   "diabetes_bloodsugar",
                   "diabetes_chol",
                   "antidepressant", 
                   "bloodpressure",
                   "ra_manage",
                   "copd_test",
                   "betablocker",
                   "bladder",
                   "falling",
                   "appeals_timely",
                   "appeals_review"],
    "2010": ["contractid",
                   "org_type",
                   "contract_name",
                   "org_marketing",
                   "breastcancer_screen",
                   "rectalcancer_screen",
                   "cv_diab_cholscreen",
                   "glaucoma_test",
                   "monitoring",
                   "flu_vaccine",
                   "pn_vaccine",
                   "physical_health",
                   "mental_health",
                   "osteo_test",
                   "physical_monitor",
                   "primaryaccess",
                   "osteo_manage",
                   "diab_healthy",
                   "bloodpressure",
                   "ra_manage",
                   "copd_test",
                   "bladder",
                   "falling",
                   "nodelays",
                   "doctor_communicate",                   
                   "carequickly",
                   "customer_service",                   
                   "overallrating_care",
                   "overallrating_plan",
                   "complaints_plan",
                   "appeals_timely",
                   "appeals_review",
                   "leave_plan",
                   "audit_problems",
                   "hold_times",
                   "info_accuracy",
                   "ttyt_available"],
    "2011": ["contractid",
                   "org_type",
                   "contract_name",
                   "org_marketing",
                   "breastcancer_screen",
                   "rectalcancer_screen",
                   "cv_cholscreen",
                   "diab_cholscreen",
                   "glaucoma_test",
                   "monitoring",
                   "flu_vaccine",
                   "pn_vaccine",
                   "physical_health",
                   "mental_health",
                   "osteo_test",
                   "physical_monitor",
                   "primaryaccess",
                   "osteo_manage",
                   "diabetes_eye",
                   "diabetes_kidney",
                   "diabetes_bloodsugar",
                   "diabetes_chol",
                   "bloodpressure",
                   "ra_manage",
                   "copd_test",
                   "bladder",
                   "falling",
                   "nodelays",
                   "doctor_communicate",
                   "carequickly",
                   "customer_service",
                   "overallrating_care",
                   "overallrating_plan",
                   "complaints_plan",
                   "appeals_timely",
                   "appeals_review",
                   "corrective_action",
                   "hold_times",
                   "info_accuracy",
                   "ttyt_available"],
    "2012": ["contractid",
                   "org_type",
                   "org_parent",
                   "org_marketing",
                   "breastcancer_screen",
                  "rectalcancer_screen",
                  "cv_cholscreen",
                  "diab_cholscreen",
                  "glaucoma_test",
                  "flu_vaccine",
                  "pn_vaccine",
                  "physical_health",
                  "mental_health",
                  "physical_monitor",
                  "primaryaccess",
                  "bmi_assess",
                  "older_medication",
                  "older_function",
                  "older_pain",
                  "osteo_manage",
                  "diabetes_eye",
                  "diabetes_kidney",
                  "diabetes_bloodsugar",
                  "diabetes_chol",
                  "bloodpressure",
                  "ra_manage",
                  "bladder",
                  "falling",
                  "readmissions",
                  "nodelays",
                  "carequickly",
                  "customer_service",
                  "overallrating_care",
                  "overallrating_plan",
                  "complaints_plan",
                  "access_problems",
                  "leave_plan",
                  "appeals_timely",
                  "appeals_review",
                  "ttyt_available"],
    "2013": ["contractid",
                   "org_type",
                   "contract_name",
                   "org_marketing",
                   "org_parent",
                  "breastcancer_screen",
                  "rectalcancer_screen",
                  "cv_cholscreen",
                  "diab_cholscreen",
                  "glaucoma_test",
                  "flu_vaccine",
                  "physical_health",
                  "mental_health",
                  "physical_monitor",
                  "bmi_assess",
                  "older_medication",
                  "older_function",
                  "older_pain",
                  "osteo_manage",
                  "diabetes_eye",
                  "diabetes_kidney",
                  "diabetes_bloodsugar",
                  "diabetes_chol",
                  "bloodpressure",
                  "ra_manage",
                  "bladder",
                  "falling",
                  "readmissions",
                  "nodelays",
                  "carequickly",
                  "customer_service",
                  "overallrating_care",
                  "overallrating_plan",
                  "coordination",
                  "complaints_plan",
                  "access_problems",
                  "leave_plan",
                  "improve",
                  "appeals_timely",
                  "appeals_review",
                  "ttyt_available",
                  "enroll_timely"],
    "2014": ["contractid",
                   "org_type",
                   "contract_name",
                   "org_marketing",
                   "org_parent",
                   "breastcancer_screen",
                   "rectalcancer_screen",
                   "cv_cholscreen",
                   "diab_cholscreen",
                   "glaucoma_test",
                   "flu_vaccine",
                   "physical_health",
                   "mental_health",
                   "physical_monitor",
                   "bmi_assess",
                   "older_medication",
                   "older_function",
                   "older_pain",
                   "osteo_manage",
                   "diabetes_eye",
                   "diabetes_kidney",
                   "diabetes_bloodsugar",
                   "diabetes_chol",
                   "bloodpressure",
                   "ra_manage",
                   "bladder",
                   "falling",
                   "readmissions",
                   "nodelays",
                   "carequickly",
                   "customer_service",
                   "overallrating_care",
                   "overallrating_plan",
                   "coordination",
                   "complaints_plan",
                   "access_problems",
                   "leave_plan",
                   "improve",
                   "appeals_timely",
                   "appeals_review",
                   "ttyt_available"],
    "2015": ["contractid",
                   "org_type",
                   "contract_name",
                   "org_marketing",
                   "org_parent",
                   "rectalcancer_screen",
                   "cv_cholscreen",
                   "diab_cholscreen",
                   "flu_vaccine",
                   "physical_health",
                   "mental_health",
                   "physical_monitor",
                   "bmi_assess",
                   "specialneeds_manage",
                   "older_medication",
                   "older_function",
                   "older_pain",
                   "osteo_manage",
                   "diabetes_eye",
                   "diabetes_kidney",
                   "diabetes_bloodsugar",
                   "diabetes_chol",
                   "bloodpressure",
                   "ra_manage",
                   "bladder",
                   "falling",
                   "readmissions",
                   "nodelays",
                   "carequickly",
                   "customer_service",
                   "overallrating_care",
                   "overallrating_plan",
                   "coordination",
                   "complaints_plan",
                   "leave_plan",
                   "improve",
                   "appeals_timely",
                   "appeals_review"],
    "2016": ["breastcancer_screen",
                   "rectalcancer_screen",
                   "flu_vaccine",
                   "physical_health",
                   "mental_health",
                   "physical_monitor",
                   "bmi_assess",
                   "specialneeds_manage",
                   "older_medication",
                   "older_function",
                   "older_pain",
                   "osteo_manage",
                   "diabetes_eye",
                   "diabetes_kidney",
                   "diabetes_bloodsugar",
                   "bloodpressure",
                   "ra_manage",
                   "falling",
                   "readmissions",
                   "nodelays",
                   "carequickly",
                   "customer_service",
                   "overallrating_care",
                   "overallrating_plan",
                   "coordination",
                   "complaints_plan",
                   "leave_plan",
                   "access_problems",
                   "improve",
                   "appeals_timely",
                   "appeals_review",
                   "ttyt_available"],
    "2017": ["breastcancer_screen",
                   "rectalcancer_screen",
                   "flu_vaccine",
                   "physical_health",
                   "mental_health",
                   "physical_monitor",
                   "bmi_assess",
                   "specialneeds_manage",
                   "older_medication",
                   "older_function",
                   "older_pain",
                   "osteo_manage",
                   "diabetes_eye",
                   "diabetes_kidney",
                   "diabetes_bloodsugar",
                   "bloodpressure",
                   "ra_manage",
                   "falling",
                   "readmissions",
                   "nodelays",
                   "carequickly",
                   "customer_service",
                   "overallrating_care",
                   "overallrating_plan",
                   "coordination",
                   "complaints_plan",
                   "leave_plan",
                   "access_problems",
                   "improve",
                   "appeals_timely",
                   "appeals_review",
                   "ttyt_available"],
    "2018": ["breastcancer_screen",
                   "rectalcancer_screen",
                   "flu_vaccine",
                   "physical_health",
                   "mental_health",
                   "physical_monitor",
                   "bmi_assess",
                   "specialneeds_manage",
                   "older_medication",
                   "older_function",
                   "older_pain",
                   "osteo_manage",
                   "diabetes_eye",
                   "diabetes_kidney",
                   "diabetes_bloodsugar",
                   "bloodpressure",
                   "ra_manage",
                   "falling",
                   "bladder",
                   "medication",
                   "readmissions",
                   "nodelays",
                   "carequickly",
                   "customer_service",
                   "overallrating_care",
                   "overallrating_plan",
                   "coordination",
                   "complaints_plan",
                   "leave_plan",
                   "access_problems",
                   "improve",
                   "appeals_timely",
                   "appeals_review",
                   "ttyt_available"]
}

# Save to pickle
with open("data/output/rating_variables.pkl", "wb") as f:
    pickle.dump(rating_vars, f)