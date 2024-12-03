import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

election_df = 'C:\\Users\\brigg\\Downloads\\a5a123.csv'
combined_theft_df = 'C:\\Users\\brigg\\Downloads\\Combined_Theft.csv'
homicide_df = 'C:\\Users\\brigg\\Downloads\\Homicide_Counts.csv'
graduation_df = 'C:\\Users\\brigg\\Downloads\\graduation.csv'

election_df = pd.read_csv(election_df)
combined_theft_df = pd.read_csv(combined_theft_df)
homicide_df = pd.read_csv(homicide_df)
graduation_df = pd.read_csv(graduation_df)

homicide_long = homicide_df.melt(id_vars=["State"], var_name="Year", value_name="Homicide_Counts")
graduation_long = graduation_df.melt(id_vars=["State"], var_name="Year", value_name="Graduation_Rates")
election_long = election_df.melt(id_vars=["State"], var_name="Year", value_name="Voting_Percentage")
combined_theft_long = combined_theft_df.melt(id_vars=["State"], var_name="Year", value_name="Theft_Cases")

for df in [homicide_long, graduation_long, election_long, combined_theft_long]:
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

homicide_long['State'] = homicide_long['State'].str.lower()
graduation_long['State'] = graduation_long['State'].str.lower()
election_long['State'] = election_long['State'].str.lower()
combined_theft_long['State'] = combined_theft_long['State'].str.lower()

dffinal = homicide_long.merge(graduation_long, on=['State', 'Year'], how='inner')
dffinal = dffinal.merge(election_long, on=['State', 'Year'], how='inner')
dffinal = dffinal.merge(combined_theft_long, on=['State', 'Year'], how='inner')  

pivoted_data = dffinal.pivot(index='Year', columns='State', values='Homicide_Counts')
#make bar chart
plt.figure(figsize=(15, 10))
plt.imshow(pivoted_data, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(label='Homicide Counts')

plt.title('Homicide Counts for Each State Across Years', fontsize=16)
plt.xlabel('States', fontsize=14)
plt.ylabel('Years', fontsize=14)
plt.xticks(ticks=range(len(pivoted_data.columns)), labels=pivoted_data.columns, rotation=90)
plt.yticks(ticks=range(len(pivoted_data.index)), labels=pivoted_data.index)

plt.tight_layout()
plt.show()
#make corellation table
correlation_data = dffinal[['Voting_Percentage', 'Homicide_Counts', 'Theft_Cases']].corr()

plt.figure(figsize=(8, 6))
plt.imshow(correlation_data, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient')

for i in range(correlation_data.shape[0]):
    for j in range(correlation_data.shape[1]):
        plt.text(j, i, f"{correlation_data.iloc[i, j]:.2f}", ha='center', va='center', color='black')

plt.xticks(ticks=range(correlation_data.shape[1]), labels=correlation_data.columns, rotation=45, ha='right')
plt.yticks(ticks=range(correlation_data.shape[0]), labels=correlation_data.index)
plt.title('Correlation Between Voting Percentage and Crime Data', fontsize=16)
plt.tight_layout()
plt.show()
#make regression
X = dffinal[['Homicide_Counts', 'Theft_Cases', 'Graduation_Rates']] 
X = sm.add_constant(X)  
y = dffinal['Voting_Percentage']  

model = sm.OLS(y, X).fit()

print(model.summary())

#creating a decision tree
dffinal.dropna(inplace=True)
X = dffinal[['Homicide_Counts', 'Theft_Cases', 'Graduation_Rates']]
y = (dffinal['Voting_Percentage'] > dffinal['Voting_Percentage'].mean()).astype(int) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Decision Tree: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
plt.figure(figsize=(15, 10))
plot_tree(
    decision_tree, 
    feature_names=list(X.columns),  # Convert to a plain list
    class_names=["Low Voting", "High Voting"], 
    filled=True, 
    rounded=True
)
plt.title("Decision Tree for Voting Percentage Analysis")
plt.show()