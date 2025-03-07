# import streamlit as st
# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# st.sidebar.title("Customer Churn Prediction")
# page = st.sidebar.radio("Choose the Menu", ["Upload & Predict", "Data Overview"])

# @st.cache_resource
# def load_pipeline():
#     with open("catboost.sav", "rb") as file:
#         pipeline = pickle.load(file)
#     return pipeline

# pipeline = load_pipeline()

# def segment_by_tenure(tenure):
#     if 0 <= tenure <= 6: return "New Customers"
#     elif 6 < tenure <= 12: return "Growing Customers"
#     elif 12 < tenure <= 36: return "Mid-term Customers"
#     elif 36 < tenure <= 72: return "Long-term Customers"
#     else: return "Unknown"

# def segment_by_totalcharges(total_charges):
#     if 0 <= total_charges <= 670: return "Low Spender"
#     elif 670 < total_charges <= 2656: return "Mid Spender"
#     elif 2656 < total_charges <= 8684.8: return "High Spender"
#     else: return "Unknown"

# def assign_customer_group(customer_segment):
#     gold = ["Long-term Customers - High Spender", 
#             "Long-term Customers - Mid Spender", 
#             "Mid-term Customers - High Spender"]
    
#     silver = ["Mid-term Customers - Mid Spender", 
#               "Growing Customers - Mid Spender"]

#     bronze = ["Mid-term Customers - Low Spender", 
#               "Growing Customers - Low Spender", 
#               "New Customers - Low Spender"]

#     if customer_segment in gold:
#         return "Gold"
#     elif customer_segment in silver:
#         return "Silver"
#     elif customer_segment in bronze:
#         return "Bronze"
#     else:
#         return "Unknown"

# if "df_pred" not in st.session_state:
#     st.session_state.df_pred = None
# if "uploaded_file" not in st.session_state:
#     st.session_state.uploaded_file = None

# if page == "Upload & Predict":
#     st.title("📊 Customer Churn Prediction for Telco Company")

#     uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

#     if uploaded_file is not None:
#         st.session_state.uploaded_file = uploaded_file

#     if st.session_state.uploaded_file is not None:
#         st.session_state.uploaded_file.seek(0)  
#         try:
#             df = pd.read_csv(st.session_state.uploaded_file, encoding="utf-8")
#             st.subheader("🔍 Data Preview")
#             st.dataframe(df.head())

#             if st.button("Predict"):
#                 with st.spinner("Predicting..."):
#                     try:
#                         expected_features = pipeline.feature_names_
#                         df_features = df[expected_features]  
#                         predictions = pipeline.predict(df_features)

#                         df['Churn_Prediction'] = predictions
#                         df['Customer_Segment'] = df.apply(lambda row: f"{segment_by_tenure(row['tenure'])} - {segment_by_totalcharges(row['TotalCharges'])}", axis=1)
#                         df['Customer_Group'] = df['Customer_Segment'].apply(assign_customer_group)

#                         churn_customers = df[df['Churn_Prediction'] == 1]
#                         churn_gold = churn_customers[churn_customers["Customer_Group"] == "Gold"]
#                         churn_silver = churn_customers[churn_customers["Customer_Group"] == "Silver"]
#                         churn_bronze = churn_customers[churn_customers["Customer_Group"] == "Bronze"]

#                         st.session_state.df_pred = df 
#                         st.session_state.churn_gold = churn_gold  
#                         st.session_state.churn_silver = churn_silver  
#                         st.session_state.churn_bronze = churn_bronze  

#                         st.success("Prediction completed!")

#                         tab1, tab2, tab3, tab4, tab5 = st.tabs([
#                             "📝 Prediction Results", 
#                             "🚨 Churn Customers", 
#                             "🏅 Gold Churn Customers", 
#                             "🥈 Silver Churn Customers", 
#                             "🥉 Bronze Churn Customers"
#                         ])

#                         with tab1:
#                             st.subheader("📝 Prediction Results")
#                             st.dataframe(df)

#                         with tab2:
#                             st.subheader("🚨 Churn Customers")
#                             st.dataframe(churn_customers)

#                         with tab3:
#                             st.subheader("🏅 Gold Churn Customers")
#                             st.dataframe(churn_gold)
#                             st.download_button("⬇ Download Gold Churn Customers", churn_gold.to_csv(index=False), "gold_churn_customers.csv", "text/csv")

#                         with tab4:
#                             st.subheader("🥈 Silver Churn Customers")
#                             st.dataframe(churn_silver)
#                             st.download_button("⬇ Download Silver Churn Customers", churn_silver.to_csv(index=False), "silver_churn_customers.csv", "text/csv")

#                         with tab5:
#                             st.subheader("🥉 Bronze Churn Customers")
#                             st.dataframe(churn_bronze)
#                             st.download_button("⬇ Download Bronze Churn Customers", churn_bronze.to_csv(index=False), "bronze_churn_customers.csv", "text/csv")

#                     except KeyError as e:
#                         st.error(f"Missing required column: {e}")
#                     except Exception as e:
#                         st.error(f"An error occurred: {e}")

#         except pd.errors.EmptyDataError:
#             st.error("⚠️ The file appears to be empty or unreadable. Check the format and try again.")
#         except Exception as e:
#             st.error(f"An error occurred while reading the file: {e}")

# elif page == "Data Overview":
#     st.title("📊 Data Overview")

#     if st.session_state.df_pred is not None:
#         df_pred = st.session_state.df_pred

#         churn_counts = df_pred["Churn_Prediction"].value_counts()
#         labels = ["Non-Churn", "Churn"]
#         colors = ["#4CAF50", "#FF5733"]

#         fig, ax = plt.subplots(figsize=(6, 6))
#         ax.pie(churn_counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, wedgeprops={"edgecolor": "black"})
#         ax.set_title("Churn Distribution")

#         st.pyplot(fig)

#         st.subheader("🏆 Customer Group Distribution")
#         group_counts = df_pred["Customer_Group"].value_counts()
#         st.bar_chart(group_counts)

#         st.subheader("📌 Customer Segmentation")
#         segment_counts = df_pred["Customer_Segment"].value_counts()
#         st.bar_chart(segment_counts)

#     else:
#         st.warning("Please upload data and run the prediction first in the 'Upload & Predict' page.")


# import streamlit as st
# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# st.sidebar.title("Customer Churn Prediction")
# page = st.sidebar.radio("Choose the Menu", ["Upload & Predict", "Data Overview"])

# @st.cache_resource
# def load_pipeline():
#     with open("catboost.sav", "rb") as file:
#         pipeline = pickle.load(file)
#     return pipeline

# pipeline = load_pipeline()

# def segment_by_tenure(tenure):
#     if 0 <= tenure <= 6: return "New Customers"
#     elif 6 < tenure <= 12: return "Growing Customers"
#     elif 12 < tenure <= 36: return "Mid-term Customers"
#     elif 36 < tenure <= 72: return "Long-term Customers"
#     else: return "Unknown"

# def segment_by_totalcharges(total_charges):
#     if 0 <= total_charges <= 670: return "Low Spender"
#     elif 670 < total_charges <= 2656: return "Mid Spender"
#     elif 2656 < total_charges <= 8684.8: return "High Spender"
#     else: return "Unknown"

# def assign_customer_group(customer_segment):
#     gold = ["Long-term Customers - High Spender", 
#             "Long-term Customers - Mid Spender", 
#             "Mid-term Customers - High Spender"]
    
#     silver = ["Mid-term Customers - Mid Spender", 
#               "Growing Customers - Mid Spender"]

#     bronze = ["Mid-term Customers - Low Spender", 
#               "Growing Customers - Low Spender", 
#               "New Customers - Low Spender"]

#     if customer_segment in gold:
#         return "Gold"
#     elif customer_segment in silver:
#         return "Silver"
#     elif customer_segment in bronze:
#         return "Bronze"
#     else:
#         return "Unknown"

# if "df_pred" not in st.session_state:
#     st.session_state.df_pred = None
# if "uploaded_file" not in st.session_state:
#     st.session_state.uploaded_file = None

# if page == "Upload & Predict":
#     st.title("📊 Customer Churn Prediction for Telco Company")

#     uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

#     if uploaded_file is not None:
#         st.session_state.uploaded_file = uploaded_file

#     if st.session_state.uploaded_file is not None:
#         st.session_state.uploaded_file.seek(0)  
#         try:
#             df = pd.read_csv(st.session_state.uploaded_file, encoding="utf-8")
#             st.subheader("🔍 Data Preview")
#             st.dataframe(df.head())

#             if st.button("Predict"):
#                 with st.spinner("Predicting..."):
#                     try:
#                         expected_features = pipeline.feature_names_
#                         df_features = df[expected_features]  
#                         predictions = pipeline.predict(df_features)

#                         df['Churn_Prediction'] = predictions
#                         df['Customer_Segment'] = df.apply(lambda row: f"{segment_by_tenure(row['tenure'])} - {segment_by_totalcharges(row['TotalCharges'])}", axis=1)
#                         df['Customer_Group'] = df['Customer_Segment'].apply(assign_customer_group)

#                         churn_customers = df[df['Churn_Prediction'] == 1]
#                         churn_gold = churn_customers[churn_customers["Customer_Group"] == "Gold"]
#                         churn_silver = churn_customers[churn_customers["Customer_Group"] == "Silver"]
#                         churn_bronze = churn_customers[churn_customers["Customer_Group"] == "Bronze"]

#                         st.session_state.df_pred = df 
#                         st.session_state.churn_gold = churn_gold  
#                         st.session_state.churn_silver = churn_silver  
#                         st.session_state.churn_bronze = churn_bronze  

#                         st.success("Prediction completed!")

#                         tab1, tab2, tab3, tab4, tab5 = st.tabs([
#                             "📝 Prediction Results", 
#                             "🚨 Churn Customers", 
#                             "🏅 Gold Churn Customers", 
#                             "🥈 Silver Churn Customers", 
#                             "🥉 Bronze Churn Customers"
#                         ])

#                         with tab1:
#                             st.subheader("📝 Prediction Results")
#                             st.dataframe(df)

#                         with tab2:
#                             st.subheader("🚨 Churn Customers")
#                             st.dataframe(churn_customers)

#                         with tab3:
#                             st.subheader("🏅 Gold Churn Customers")
#                             st.dataframe(churn_gold)
#                             st.download_button("⬇ Download Gold Churn Customers", churn_gold.to_csv(index=False), "gold_churn_customers.csv", "text/csv")

#                         with tab4:
#                             st.subheader("🥈 Silver Churn Customers")
#                             st.dataframe(churn_silver)
#                             st.download_button("⬇ Download Silver Churn Customers", churn_silver.to_csv(index=False), "silver_churn_customers.csv", "text/csv")

#                         with tab5:
#                             st.subheader("🥉 Bronze Churn Customers")
#                             st.dataframe(churn_bronze)
#                             st.download_button("⬇ Download Bronze Churn Customers", churn_bronze.to_csv(index=False), "bronze_churn_customers.csv", "text/csv")

#                     except KeyError as e:
#                         st.error(f"Missing required column: {e}")
#                     except Exception as e:
#                         st.error(f"An error occurred: {e}")

#         except pd.errors.EmptyDataError:
#             st.error("⚠️ The file appears to be empty or unreadable. Check the format and try again.")
#         except Exception as e:
#             st.error(f"An error occurred while reading the file: {e}")

# elif page == "Data Overview":
#     st.title("📊 Data Overview")

#     if st.session_state.df_pred is not None:
#         df_pred = st.session_state.df_pred

#         churn_counts = df_pred["Churn_Prediction"].value_counts()
#         labels = ["Non-Churn", "Churn"]
#         colors = ["#4CAF50", "#FF5733"]

#         fig, ax = plt.subplots(figsize=(6, 6))
#         ax.pie(churn_counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, wedgeprops={"edgecolor": "black"})
#         ax.set_title("Churn Distribution")

#         st.pyplot(fig)

#         st.subheader("🏆 Customer Group Distribution")
#         group_counts = df_pred["Customer_Group"].value_counts()
#         st.bar_chart(group_counts)

#         st.subheader("📌 Customer Segmentation")
#         segment_counts = df_pred["Customer_Segment"].value_counts()
#         st.bar_chart(segment_counts)

#     else:
#         st.warning("Please upload data and run the prediction first in the 'Upload & Predict' page.")


import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.sidebar.title("Customer Churn Prediction")
page = st.sidebar.radio("Choose the Menu", ["Upload & Predict", "Data Overview"])

@st.cache_resource
def load_pipeline():
    with open("catboost.sav", "rb") as file:
        pipeline = pickle.load(file)
    return pipeline

pipeline = load_pipeline()

def segment_by_tenure(tenure):
    if 0 <= tenure <= 6: return "New Customers"
    elif 6 < tenure <= 12: return "Growing Customers"
    elif 12 < tenure <= 36: return "Mid-term Customers"
    elif 36 < tenure <= 72: return "Long-term Customers"
    else: return "Unknown"

def segment_by_totalcharges(total_charges):
    if 0 <= total_charges <= 670: return "Low Spender"
    elif 670 < total_charges <= 2656: return "Mid Spender"
    elif 2656 < total_charges <= 8684.8: return "High Spender"
    else: return "Unknown"

def assign_customer_group(customer_segment):
    gold = ["Long-term Customers - High Spender", 
            "Long-term Customers - Mid Spender", 
            "Mid-term Customers - High Spender"]
    
    silver = ["Mid-term Customers - Mid Spender", 
              "Growing Customers - Mid Spender"]

    bronze = ["Mid-term Customers - Low Spender", 
              "Growing Customers - Low Spender", 
              "New Customers - Low Spender"]

    if customer_segment in gold:
        return "Gold"
    elif customer_segment in silver:
        return "Silver"
    elif customer_segment in bronze:
        return "Bronze"
    else:
        return "Unknown"

if "df_pred" not in st.session_state:
    st.session_state.df_pred = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if page == "Upload & Predict":
    st.title("📊 Customer Churn Prediction for Telco Company")

    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    if st.session_state.uploaded_file is not None:
        st.session_state.uploaded_file.seek(0)  
        try:
            df = pd.read_csv(st.session_state.uploaded_file, encoding="utf-8")
            st.subheader("🔍 Data Preview")
            st.dataframe(df.head())

            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    try:
                        expected_features = pipeline.feature_names_
                        df_features = df[expected_features]  
                        predictions = pipeline.predict(df_features)

                        df['Churn_Prediction'] = predictions
                        df['Customer_Segment'] = df.apply(lambda row: f"{segment_by_tenure(row['tenure'])} - {segment_by_totalcharges(row['TotalCharges'])}", axis=1)
                        df['Customer_Group'] = df['Customer_Segment'].apply(assign_customer_group)

                        churn_customers = df[df['Churn_Prediction'] == 1]
                        churn_gold = churn_customers[churn_customers["Customer_Group"] == "Gold"]
                        churn_silver = churn_customers[churn_customers["Customer_Group"] == "Silver"]
                        churn_bronze = churn_customers[churn_customers["Customer_Group"] == "Bronze"]

                        st.session_state.df_pred = df 
                        st.session_state.churn_gold = churn_gold  
                        st.session_state.churn_silver = churn_silver  
                        st.session_state.churn_bronze = churn_bronze  

                        st.success("Prediction completed!")

                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "📝 Prediction Results", 
                            "🚨 Churn Customers", 
                            "🏅 Gold Churn Customers", 
                            "🥈 Silver Churn Customers", 
                            "🥉 Bronze Churn Customers"
                        ])

                        with tab1:
                            st.subheader("📝 Prediction Results")
                            st.dataframe(df)

                        with tab2:
                            st.subheader("🚨 Churn Customers")
                            st.dataframe(churn_customers)

                        with tab3:
                            st.subheader("🏅 Gold Churn Customers")
                            st.dataframe(churn_gold)
                            st.download_button("⬇ Download Gold Churn Customers", churn_gold.to_csv(index=False), "gold_churn_customers.csv", "text/csv")

                        with tab4:
                            st.subheader("🥈 Silver Churn Customers")
                            st.dataframe(churn_silver)
                            st.download_button("⬇ Download Silver Churn Customers", churn_silver.to_csv(index=False), "silver_churn_customers.csv", "text/csv")

                        with tab5:
                            st.subheader("🥉 Bronze Churn Customers")
                            st.dataframe(churn_bronze)
                            st.download_button("⬇ Download Bronze Churn Customers", churn_bronze.to_csv(index=False), "bronze_churn_customers.csv", "text/csv")

                    except KeyError as e:
                        st.error(f"Missing required column: {e}")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        except pd.errors.EmptyDataError:
            st.error("⚠️ The file appears to be empty or unreadable. Check the format and try again.")
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

elif page == "Data Overview":
    st.title("📊 Data Overview")

    if st.session_state.df_pred is not None:
        df_pred = st.session_state.df_pred

        tab1, tab2, tab3 = st.tabs([
            "📊 Churn Distribution", 
            "🏆 Customer Group Distribution", 
            "📌 Customer Segmentation"
        ])

        with tab1:
            churn_counts = df_pred["Churn_Prediction"].value_counts()
            labels = ["Non-Churn", "Churn"]
            colors = ["#4CAF50", "#FF5733"]

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.pie(churn_counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, wedgeprops={"edgecolor": "black"})
            ax.set_title("Churn Distribution", fontsize=10, fontweight="bold")
            st.pyplot(fig)

        with tab2:
            st.subheader("🏆 Customer Group Distribution")
            group_counts = df_pred["Customer_Group"].value_counts()

            fig, ax = plt.subplots(figsize=(5, 3))
            group_counts.plot(kind="bar", color="#87CEFA", edgecolor="black", ax=ax)

            ax.set_title("Customer Group Distribution", fontsize=10, fontweight="bold")
            ax.set_xlabel("Customer Group", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.tick_params(axis='x', rotation=0, labelsize=8, pad=5)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            st.pyplot(fig)

        with tab3:
            st.subheader("📌 Customer Segmentation")
            segment_counts = df_pred["Customer_Segment"].value_counts()

            fig, ax = plt.subplots(figsize=(6, 3))
            segment_counts.plot(kind="bar", color="#FFB6C1", edgecolor="black", ax=ax)

            ax.set_title("Customer Segmentation", fontsize=10, fontweight="bold")
            ax.set_xlabel("Customer Segment", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.tick_params(axis='x', rotation=90, labelsize=7, pad=5)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            st.pyplot(fig)

    else:
        st.warning("Please upload data and run the prediction first in the 'Upload & Predict' page.")

# import streamlit as st
# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# st.sidebar.title("Customer Churn Prediction")
# page = st.sidebar.radio("Choose the Menu", ["Upload & Predict", "Data Overview"])

# @st.cache_resource
# def load_pipeline():
#     with open("catboost.sav", "rb") as file:
#         pipeline = pickle.load(file)
#     return pipeline

# pipeline = load_pipeline()

# def segment_by_tenure(tenure):
#     if 0 <= tenure <= 6: return "New Customers"
#     elif 6 < tenure <= 12: return "Growing Customers"
#     elif 12 < tenure <= 36: return "Mid-term Customers"
#     elif 36 < tenure <= 72: return "Long-term Customers"
#     else: return "Unknown"

# def segment_by_totalcharges(total_charges):
#     if 0 <= total_charges <= 670: return "Low Spender"
#     elif 670 < total_charges <= 2656: return "Mid Spender"
#     elif 2656 < total_charges <= 8684.8: return "High Spender"
#     else: return "Unknown"

# def assign_customer_group(customer_segment):
#     gold = ["Long-term Customers - High Spender", "Long-term Customers - Mid Spender", "Mid-term Customers - High Spender"]
#     silver = ["Mid-term Customers - Mid Spender", "Growing Customers - Mid Spender"]
#     bronze = ["Mid-term Customers - Low Spender", "Growing Customers - Low Spender", "New Customers - Low Spender"]

#     if customer_segment in gold:
#         return "Gold"
#     elif customer_segment in silver:
#         return "Silver"
#     elif customer_segment in bronze:
#         return "Bronze"
#     else:
#         return "Unknown"

# if "df_pred" not in st.session_state:
#     st.session_state.df_pred = None

# if page == "Upload & Predict":
#     st.title("📊 Customer Churn Prediction for Telco Company")

#     uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
#             st.subheader("🔍 Data Preview")
#             st.dataframe(df.head())

#             if st.button("Predict"):
#                 with st.spinner("Predicting..."):
#                     try:
#                         df['Churn_Prediction'] = pipeline.predict(df[pipeline.feature_names_])
#                         df['Customer_Segment'] = df.apply(lambda row: f"{segment_by_tenure(row['tenure'])} - {segment_by_totalcharges(row['TotalCharges'])}", axis=1)
#                         df['Customer_Group'] = df['Customer_Segment'].apply(assign_customer_group)

#                         st.session_state.df_pred = df  

#                         st.success("Prediction completed!")

#         except Exception as e:
#             st.error(f"An error occurred: {e}")

# elif page == "Data Overview":
#     st.title("📊 Data Overview")

#     if st.session_state.df_pred is not None:
#         df_pred = st.session_state.df_pred

#         st.subheader("📌 Customer Segmentation")
#         segment_counts = df_pred["Customer_Segment"].value_counts()

#         fig, ax = plt.subplots(figsize=(7, 4))
#         segment_counts.plot(kind="bar", color="#FFB6C1", edgecolor="black", ax=ax)

#         ax.set_title("Customer Segmentation", fontsize=10, fontweight="bold")
#         ax.set_xlabel("Customer Segment", fontsize=9)
#         ax.set_ylabel("Count", fontsize=9)

#         # **PERBAIKAN: Pastikan jumlah tick sesuai jumlah kategori**
#         ax.set_xticks(range(len(segment_counts)))
#         ax.set_xticklabels(segment_counts.index, rotation=25, ha="right", fontsize=7)

#         ax.grid(axis="y", linestyle="--", alpha=0.7)

#         st.pyplot(fig)

#         st.subheader("🏅 Customer Group Distribution")
#         group_counts = df_pred["Customer_Group"].value_counts()

#         fig, ax = plt.subplots(figsize=(5, 4))
#         group_counts.plot(kind="bar", color=["#FFD700", "#C0C0C0", "#CD7F32"], edgecolor="black", ax=ax)

#         ax.set_title("Customer Group Distribution", fontsize=10, fontweight="bold")
#         ax.set_xlabel("Customer Group", fontsize=9)
#         ax.set_ylabel("Count", fontsize=9)

#         ax.set_xticks(range(len(group_counts)))
#         ax.set_xticklabels(group_counts.index, rotation=0, fontsize=9)

#         ax.grid(axis="y", linestyle="--", alpha=0.7)

#         st.pyplot(fig)

#         st.subheader("📋 Churn & Customer Group Breakdown")
#         tab1, tab2, tab3, tab4 = st.tabs(["All Customers", "Gold Customers", "Silver Customers", "Bronze Customers"])

#         with tab1:
#             st.dataframe(df_pred)

#         with tab2:
#             gold_customers = df_pred[df_pred["Customer_Group"] == "Gold"]
#             st.dataframe(gold_customers)
#             st.download_button("⬇ Download Gold Customers", gold_customers.to_csv(index=False), "gold_customers.csv", "text/csv")

#         with tab3:
#             silver_customers = df_pred[df_pred["Customer_Group"] == "Silver"]
#             st.dataframe(silver_customers)
#             st.download_button("⬇ Download Silver Customers", silver_customers.to_csv(index=False), "silver_customers.csv", "text/csv")

#         with tab4:
#             bronze_customers = df_pred[df_pred["Customer_Group"] == "Bronze"]
#             st.dataframe(bronze_customers)
#             st.download_button("⬇ Download Bronze Customers", bronze_customers.to_csv(index=False), "bronze_customers.csv", "text/csv")

#     else:
#         st.warning("Please upload data and run the prediction first in the 'Upload & Predict' page.")

