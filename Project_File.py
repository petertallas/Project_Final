import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

social_media1 = "social_media_usage.csv"

s = pd.read_csv(social_media1)

def clean_sm(x):
    return np.where(x==1, 1, 0)

s['sm_li']=clean_sm(s['web1h'])

s1 = s[['sm_li','income','educ2', 'par', 'marital', 'gender', 'age']]

s2 = s1[
    (s1["income"] < 10) &
    (s1["educ2"] < 9) &
    (s1["age"] < 99)
]


s2 = s2.copy()

s2.loc[:, "married"] = np.where(s2["marital"] == 1, 1, 0)
s2.loc[:, "female"] = np.where(s2["gender"] == 2, 1, 0)
s2.loc[:, "parent"] = np.where(s2["par"] == 1, 1, 0)

ss = s2[['sm_li','income','educ2', 'parent', 'married', 'female', 'age']]

ss = ss.dropna()

y = s2["sm_li"]

X = s2[["income", "educ2", "parent", "married", "female", "age"]]


X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   stratify=y,
                                                   test_size=0.2,
                                                   random_state=987)



lr = LogisticRegression(class_weight='balanced')

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)


##-----##


st.title("""
         
Are you a LinkedIn User?
         
## Please complete the fields below, and we can predict whether or not you use LinkedIn:
 
""")
 
# Income
income_options = {
    "Less than $10k": 1,
    "$10k to under $20k": 2,
    "$20k to under $30k": 3,
    "$30k to under $40k": 4,
    "$40k to under $50k": 5,
    "$50k to under $75k": 6,
    "$75k to under $100k": 7,
    "$100k to under $150k": 8,
    "$150k or more": 9
}
 
income_key = st.selectbox("Income Level", list(income_options.keys()))
income_value = income_options[income_key]
 
# Education
education_options = {
    "Less than high school (Grades 1-8 or no formal schooling)": 1,
    "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)": 2,
    "High school graduate (Grade 12 with diploma or GED certificate)": 3,
    "Some college, no degree (includes some community college)": 4,
    "Two-year associate degree from a college or university": 5,
    "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)": 6,
    "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)": 7,
    "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)": 8
}
 
education_key = st.selectbox("Education Level", list(education_options.keys()))
education_value = education_options[education_key]
 
# Parent?
parent_options = {
    "Yes": 1,
    "No": 0
}
on = st.toggle('Parent')
if on:
    parent_value = 1
else:
    parent_value = 0
 
# Married
married_options = {
    "Yes": 1,
    "No": 0
}

married_selected = st.checkbox("Married") 
if married_selected:
    married_value= 1
else:
    married_value= 0
 
# Gender
gender_options = {
    "Male": 0,
    "Female": 1
}
 
gender_key = st.selectbox("Gender", list(gender_options.keys()))
gender_value = gender_options[gender_key]
 
# Age
age_value = st.slider("Age", min_value = 0, max_value = 98, value = 10, step = None)
 
user_input = pd.DataFrame({
    "income": [income_value],
    "educ2": [education_value],
    "parent": [parent_value],
    "married": [married_value],
    "female": [gender_value],
    "age": [age_value]
})
 
user_result = lr.predict(user_input)
user_prob = lr.predict_proba(user_input)
 
if user_result == 1:
    st.markdown("### Yes! You are most likely a Linkedin User.")
else:
    st.markdown("### No, you likely do not use LinkedIn.")
 
st.markdown(f"#### Probability you are a LinkedIn user: {user_prob[0][1]}")










