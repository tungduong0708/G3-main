import pandas as pd

# Đọc file CSV
df = pd.read_csv(r"C:\Users\tungd\OneDrive - MSFT\Second Year\ML\ACMMM25 - Grand Challenge on Multimedia Verification\G3-main\data\im2gps3k\15_llm_predict_results_rag.csv")

# Xóa 4 cột cuối
df = df.iloc[:, :-2]

# Ghi lại nếu muốn
df.to_csv(r"C:\Users\tungd\OneDrive - MSFT\Second Year\ML\ACMMM25 - Grand Challenge on Multimedia Verification\G3-main\data\im2gps3k\15_llm_predict_results_rag.csv", index=False)
