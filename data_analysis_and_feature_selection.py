#Feature selection 1

#Convert non numeric values in numeric values
df= df.apply(pd.to_numeric, errors='coerce')

#Check datatypes
print("\n",df.info())

#Exclude null columns
df_cl= df.drop(columns=['name', 'MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','HNR','spread1','D2'])
df_cl






#Data analysis

#Summary Statistics: Understand the distribution of each feature (mean, median, std, etc.).

#Summary statistics for the dataset
print("\nSummary statistics for the whole dataset:\n ", df_cl.describe())
print("\n------------------------------------------------------------------------------------------------------------------------------------------\n")

#General correlation matrix
corr_matrix_general=df_cl.corr()
print("\nCorrelation matrix of the whole dataset\n",corr_matrix_general)

#Visualization
plt.figure(figsize=(7, 6))
sns.heatmap(corr_matrix_general, annot=True)
plt.title("\nGraph A: Heatmap of the whole dataset correlation matrix\n",fontsize=15)
plt.show()

#Summary statistics for each column and visualization
columns_to_process = df_cl.columns[0:17]
for col in columns_to_process:
    print(f"\nSummary statistics for the {col}:")
    print(df_cl[col].describe())
    
 #mean
    col_mean = df_cl[col].mean()
    print(f"\nMean of {col}: {col_mean}")
    plt.figure(figsize=(8, 6))
    plt.hist(df_cl[col], bins=20, color='green', edgecolor='black')
    plt.axvline(col_mean, color='orange', linestyle='--', label=f'Mean = {col_mean:.2f}')
    plt.title(f"Mean of {col}", fontsize=8)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend(fontsize=6)
    plt.show()    

#median
    col_median = df_cl[col].median()
    print(f"Median of {col}: {col_median}")
    plt.figure(figsize=(8, 6))
    plt.hist(df_cl[col], bins=20, color='green', edgecolor='black')
    plt.axvline(col_median, color='orange', linestyle='--', label=f'Median = {col_median:.2f}')
    plt.title(f"Median of {col}", fontsize=8)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend(fontsize=6)
    plt.show()
    
 #std
    col_std = df_cl[col].std()
    print(f"Standard deviation of {col}: {col_std}")
    plt.figure(figsize=(8, 6))
    plt.hist(df_cl[col], bins=20, color='green', edgecolor='black')
    plt.axvline(col_std, color='orange', linestyle='--', label=f'Std Dev = {col_std:.2f}')
    plt.title(f"Standard Deviation of {col}", fontsize=8)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend(fontsize=6)
    plt.show()
    
  #min/max
    col_min = df_cl[col].min()
    col_max = df_cl[col].max()
    print(f"Minimum of {col}: {col_min}")
    print(f"Maximum of {col}: {col_max}")
    plt.figure(figsize=(8, 6))
    plt.hist(df_cl[col], bins=20, color='green', edgecolor='black', label='Data Distribution')
    plt.axvline(col_min, color='red', linestyle='dashed', linewidth=2, label='Min Value')
    plt.axvline(col_max, color='blue', linestyle='dashed', linewidth=2, label='Max Value')
    plt.title(f'Histogram with Min and Max Values of {col}', fontsize=8)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend(fontsize=6)
    plt.show()
    
 #Correlation matrix with the 'status' column and visualization
    if 'status' in df_cl.columns:
        correlation_matrix = df_cl[[col, 'status']].corr()
        print(f"\nCorrelation matrix of {col} with 'status':\n", correlation_matrix)
        plt.figure(figsize=(4, 4))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'Correlation Matrix: {col} vs Status', fontsize=12)
        plt.show()
    print("\n--------------------------------------------------------------------------------------------------------------------------------------\n")




#Feature selection 2

#Visualization of the degree of correlation of each column with the status column
correlations_with_status = df_cl.corr()['status'].sort_values(ascending=False)

#Drop the 'status' column
correlations_with_status = correlations_with_status.drop('status')

#Visualization
plt.figure(figsize=(10, 8))
sns.heatmap(correlations_with_status.to_frame(), annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title("Graph B: Correlation of All Columns with 'Status'", fontsize=14)
plt.ylabel('Features')
plt.xlabel('Correlation with Status')
plt.show()

#Drop columns whose correlation with the 'satus' column is under the trashold of 15%
df_cl= df_cl.drop(columns=['MDVP:Shimmer(dB)'])
df_cl

#Visualize the corellation of all the columns (except satus) with eachother
df_cl_no_status= df_cl.drop(columns=['status'])
df_cl_matrix_features=df_cl_no_status.corr()

#Visualization
plt.figure(figsize=(10, 8))
sns.heatmap(df_cl_matrix_features, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title("Graph C: Correlation of all the columns with eachother", fontsize=14)
plt.show()

#Define the threshold
threshold = 0.95

#Find columns to drop based on threshold
#Keep only columns where the max correlation (excluding diagonal) is below the threshold
mask = (df_cl_matrix_features.abs() > threshold).sum() > 1
columns_to_drop = mask[mask].index

#Drop columns and rows from the correlation matrix
filtered_corr_matrix = df_cl_matrix_features.drop(columns=columns_to_drop, index=columns_to_drop)

print("\nFiltered Correlation Matrix:")
print(filtered_corr_matrix,"\n")

#Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(filtered_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title("Graph D: Correlation of all the columns with eachother, filtered", fontsize=11)
plt.show()

print("\n---------------------------------------------------------------------------------------------------------------------------------------\n")

#Which columns are highly correlated with which columns
high_corr_pairs = []
for col in df_cl_matrix_features.columns:
    for row in df_cl_matrix_features.index:
        # Exclude the diagonal correlation (with itself)
        if col != row and abs(df_cl_matrix_features.loc[row, col]) > threshold:
            high_corr_pairs.append((row, col, df_cl_matrix_features.loc[row, col]))

#Display high correlation pairs
print("\nColumns with correlations higher than the threshold:")
for pair in high_corr_pairs:
    print(f"{pair[0]} and {pair[1]}: Correlation = {pair[2]:.2f}")

#The values "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP" and "MDVP:Jitter(%)", and the values "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:DDA" and "MDVP:Shimmer" are highly correlated between eachother (over the 95% thrashold); therefore, to avoid redundancy, we will only keeps for each of the two groups the values with the higher correlation with the "status" column: "MDVP:PPQ" and "MDVP:Shimmer".
#The values "MDVP:Shimmer" and "MDVP:APQ" are exactly at the 95% thrashold, therefore we could dump one of the two, however, "MDVP:APQ" is not as highly correlated with the other values of the same group, therfore we chose tho leave it
df_cl= df_cl.drop(columns=["MDVP:RAP","Jitter:DDP", "MDVP:Jitter(%)","Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:DDA"])
df_cl





#Normalization
#Apply Min-Max Scaling to each column
df_n = (df_cl - df_cl.min()) / (df_cl.max() - df_cl.min())
#Print normalized dataset
df_n
